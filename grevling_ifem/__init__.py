from __future__ import annotations

from pathlib import Path
from typing import Any, IO
from io import StringIO
import netCDF4 as nc
import click
import h5py
from lxml import etree

import numpy as np
import scipy.io
from tqdm import tqdm

from grevling import Case, api
from grevling.parameters import GradedParameter, UniformParameter
from pydantic import BaseModel
from splipy.io import G2
from splipy import SplineObject
from splipy.utils import section_from_index


class G2Object(G2):
    """G2 reader subclass to allow reading from a stream."""

    def __init__(self, fstream: IO, mode: str):
        self.fstream = fstream
        self.onlywrite = mode == "w"
        super(G2Object, self).__init__("")

    def __enter__(self) -> "G2Object":
        return self


class TopologySet:
    objects: list[tuple[int, tuple[int, ...]]]
    slaves: list[TopologySet]

    def __init__(self):
        self.objects = [[], [], [], []]
        self.slaves = []

    def parse_set(self, ndims: int, elt):
        pardim = {
            "vertex": 0,
            "edge": 1,
            "face": 2,
        }[elt.attrib["type"]]
        for item in elt.findall("./item"):
            patchid = int(item.attrib["patch"])
            for index in item.text.split():
                section = tuple(section_from_index(ndims, pardim, int(index) - 1))
                self.objects[pardim].append((patchid, section))

    def bind(self, slave: "TopologySet"):
        self.slaves.append(slave)

    def root_parts(self, pardim: int):
        yield from self.objects[pardim]

    def parts(self, pardim: int):
        yield from self.root_parts(pardim)
        for slave in self.slaves:
            yield from slave.parts(pardim)


parameter_dtype = np.dtype(
    [
        ("name", "S20"),
        ("min", float),
        ("max", float),
        ("unit", "S20"),
    ]
)


class Config(BaseModel):
    xinpname: str
    h5name: str
    basisname: str
    fieldname: str
    matrixname: str
    rhsname: str
    ncomps: int
    ndims: int


def load_patches(cfg: Config, f: h5py.File) -> list[SplineObject]:
    patches = []
    grp = f[f"0/{cfg.basisname}/basis"]
    for i in range(len(grp)):
        patchdata = StringIO(grp[str(i + 1)][:].tobytes().decode())
        with G2Object(patchdata, "r") as f:
            patches.append(f.read()[0])
    return patches


def load_numbering(
    cfg: Config, f: h5py.File, patches: list[SplineObject]
) -> list[np.ndarray]:
    patch_lengths = [len(patch) for patch in patches]
    group = f[f"0/{cfg.basisname}/l2g-node"]
    return [group[str(i + 1)][: patch_lengths[i]] for i in range(len(group))]


def load_xml(path: Path) -> etree.ElementTree:
    with open(path, "r") as f:
        xml = etree.parse(f)

    while True:
        include = xml.find(".//include")
        if include is None:
            break

        parser = etree.XMLParser(recover=True)
        with open(path.parent / include.text) as f:
            string = "<temp>" + f.read() + "</temp>"
            subxml = etree.fromstring(string, parser)

        insertpoint = include
        for node in subxml.getchildren():
            insertpoint.addnext(node)
            insertpoint = node

        include.getparent().remove(include)

    return xml


def load_dirichlet(cfg: Config, xml: etree.ElementTree):
    toposets = {}
    for toposet in xml.findall(".//topologysets/set"):
        name = toposet.attrib["name"]
        toposets.setdefault(name, TopologySet())
        toposets[name].parse_set(cfg.ndims, toposet)

    for rigid in xml.findall(".//rigid"):
        toposets[rigid.attrib["master"]].bind(toposets[rigid.attrib["slave"]])

    for dirichlet in xml.findall(".//dirichlet"):
        toposet = toposets[dirichlet.attrib["set"]]
        comps = tuple(
            int(c) - 1 for c in dirichlet.attrib["comp"] if int(c) <= cfg.ndims
        )
        if not comps:
            continue
        for ndim in range(0, cfg.ndims):
            for patchid, section in toposet.parts(ndim):
                yield patchid, section, comps


def load_nodal_classification(
    cfg: Config,
    xml: etree.ElementTree,
    patches: list[SplineObject],
    numbering: list[np.ndarray],
):
    ndofs = max(max(n) for n in numbering)
    mask = np.ones((ndofs * cfg.ncomps,), dtype=bool)

    for patchid, section, comps in load_dirichlet(cfg, xml):
        patch = patches[patchid - 1]
        patch.set_dimension(1)
        data = numbering[patchid - 1]
        data = data.reshape(*patch.shape, 1, order="F")
        patch.controlpoints = data
        sec = patch.section(*section, unwrap_points=False)
        inds = sec.controlpoints.flatten() - 1
        inds *= cfg.ncomps
        for comp in comps:
            mask[inds + comp] = False

    return mask


def load_field(cfg: Config, f: h5py.File, numbering: list[np.ndarray]) -> np.ndarray:
    maxn = max(max(n) for n in numbering)
    retval = None

    group = f[f"0/{cfg.basisname}/fields/{cfg.fieldname}"]
    for i, n in enumerate(numbering):
        data = group[str(i + 1)][:]
        data = data.reshape(len(n), -1)
        if retval is None:
            ncomps = data.shape[-1]
            retval = np.empty((maxn, ncomps), dtype=float)
        retval[n - 1, :] = data

    assert retval is not None
    return retval.flatten()


def extract_samples(case: Case, cfg: Config, target: Path) -> None:
    dataset = nc.Dataset(target, "w")
    dataset.Conventions = "DTHORSAMPLE-0.1"
    dataset.createDimension("PARAMETER", len(case.parameters))
    dataset.createDimension("SAMPLE")

    # Write parameter space
    parameter_space = np.zeros((len(case.parameters),), dtype=parameter_dtype)
    for i, (name, values) in enumerate(case.parameters.items()):
        assert isinstance(values, (GradedParameter, UniformParameter))
        parameter_space[i] = (name, values[0], values[-1], "")

    parameter_dtype_nc = dataset.createCompoundType(parameter_dtype, "parameter")
    var = dataset.createVariable("parameter_space", parameter_dtype_nc, ("PARAMETER",))
    var[:] = parameter_space
    var.type = "TENSOR"

    parameter_names = list(case.parameters.keys())

    # Write sampling scheme
    samples = []
    for instance in case.instances(api.Status.Downloaded):
        ctx = instance.context
        samples.append([ctx[p] for p in parameter_names])
    samples = np.array(samples, dtype=float)
    var = dataset.createVariable("sample", samples.dtype, ("SAMPLE", "PARAMETER"))
    var[:] = samples

    # Load some necessary data from one single instance (assumed to be representative)
    instance = next(case.instances(api.Status.Downloaded))
    logpath = case.storagepath / instance.logdir
    h5_path = logpath / cfg.h5name

    xinp = load_xml(logpath / cfg.xinpname)
    with h5py.File(h5_path, "r") as f:
        patches = load_patches(cfg, f)
        numbering = load_numbering(cfg, f, patches)
    classification = load_nodal_classification(cfg, xinp, patches, numbering)

    dataset.createDimension("NODE", len(classification))
    dataset.createDimension("DOF", sum(classification))
    dataset.createDimension("DOU", sum(~classification))

    # Write nodal classification
    var = dataset.createVariable("nodemask", "i1", ("NODE",))
    var[:] = classification

    # Prepare datasets and types
    sol = dataset.createVariable("sol", float, ("SAMPLE", "NODE"))
    sol.lifted = "TRUE"
    sol.lift = "lift_sol"

    lift = dataset.createVariable("lift_sol", float, ("SAMPLE", "NODE"))
    lift.is_lift = "TRUE"
    lift.vector = "sol"

    rhs = dataset.createVariable("rhs", float, ("SAMPLE", "DOF"))

    dataset.createDimension("NNZ")
    lhs_dtype = np.dtype([("i", int), ("j", int), ("v", float)])
    lhs_dtype_nc = dataset.createCompoundType(lhs_dtype, "ijv")
    lhs = dataset.createVariable("lhs", lhs_dtype_nc, ("SAMPLE", "NNZ"))
    lhs.space = "DOF"

    # Extract data from instances
    for i, instance in tqdm(
        enumerate(case.instances(api.Status.Downloaded)), "Extracting"
    ):
        logpath = case.storagepath / instance.logdir
        h5_path = logpath / cfg.h5name

        with h5py.File(h5_path, "r") as f:
            u = load_field(cfg, f, numbering)
            t = u.copy()
            t[classification] = 0
            sol[i, :] = u
            lift[i,] = t

        f = scipy.io.mmread(logpath / cfg.rhsname).flatten()
        rhs[i, :] = f

        mx = scipy.io.mmread(logpath / cfg.matrixname).asformat("coo")
        lhs[i, :] = list(zip(mx.row, mx.col, mx.data))


class Plugin(api.Plugin):
    config: Config

    def __init__(self, case: Case, settings: Any) -> None:
        self.config = Config.model_validate(settings)

    def commands(self, ctx: click.Context) -> list[click.Command]:
        cs: Case = ctx.obj["case"]

        @click.command("samples")
        @click.argument(
            "target",
            default="samples.nc",
            type=click.Path(path_type=Path, dir_okay=False, writable=True),
        )
        def samples(target: Path) -> None:
            extract_samples(cs, self.config, target)

        return [samples]
