from pyhdf.HDF import HDF, HC, HDF4Error
from pyhdf.SD import SD
import pyhdf.VS
import pyhdf.V

import numpy as np
import xarray as xr

def rootgroups(v):
    groups = set()
    has_parent = set()
    ref = -1
    while True:
        try:
            ref = v.getid(ref)
        except HDF4Error:
            break
        vg = v.attach(ref)
        tag = vg._tag
        groups.add(ref)
        has_parent |= set(ref for _, ref in vg.tagrefs())
        vg.detach()
    return groups - has_parent

class HDFGroup(object):
    kind = "group"
    def __init__(self, hdf, ref):
        vg = hdf.v.attach(ref)
        self.gclass = vg._class
        self.name = vg._name
        self.children = [c for c in (decode_ref(hdf, ref, tag) for tag, ref in vg.tagrefs()) if c is not None]
        vg.detach()

    def close(self):
        for child in self.children:
            child.close()

    def __str__(self):
        return "{} {} {}".format(self.kind, self.gclass, self.name)

class HDFScientificData(object):
    kind = "scientific data"
    def __init__(self, hdf, ref):
        self.hdf = hdf
        self.index = self.hdf.sd.reftoindex(ref)
        sds = self.hdf.sd.select(self.index)
        self.name, self.rank, self.dims, self.dtype, self.nattrs = sds.info()
        self.dimnames = [sds.dim(i).info()[0] for i in range(self.rank)]
        sds.endaccess()

    def __getitem__(self, index):
        sds = self.hdf.sd.select(self.index)
        res = sds[index]
        sds.endaccess()
        return res

    def close(self):
        pass

    def __str__(self):
        return "{} {} {} {}".format(self.kind, self.name, self.dims, self.dimnames)

class HDFVData(object):
    kind = "vdata"
    def __init__(self, hdf, ref):
        vd = hdf.vs.attach(ref)
        nrecs, intmode, self.fields, size, self.name = vd.inquire()
        try:
            self.data = vd[:]
        except HDF4Error:
            self.data = []
        vd.detach()

    def close(self):
        pass

    def __str__(self):
        return "{} {} {}".format(self.kind, self.name, ", ".join(self.fields))

def decode_ref(hdf, ref, tag=None):
    if tag is None:
        vg = hdf.v.attach(ref)
        tag = vg._tag
        vg.detach()
    if tag == HC.DFTAG_VH:
        return HDFVData(hdf, ref)
    elif tag == HC.DFTAG_NDG:
        return HDFScientificData(hdf, ref)
    elif tag == HC.DFTAG_VG:
        return HDFGroup(hdf, ref)
    else:
        pass

class HDFReader(object):
    kind = "file"
    def __init__(self, filename):
        self.name = filename
        self.hdf = HDF(filename)
        self.sd = SD(filename)
        self.vs = self.hdf.vstart()
        self.v = self.hdf.vgstart()
        self.root_refs = rootgroups(self.v)
        self.children = [decode_ref(self, ref) for ref in self.root_refs]

    def close(self):
        for child in self.children:
            child.close()
        self.v.end()
        self.vs.end()
        self.sd.end()
        self.hdf.close()

    def __del__(self):
        self.close()

    def __str__(self):
        return "{} {}".format(self.kind, self.name)

def print_tree(obj, level=0):
    print("  "*level + str(obj))
    for child in getattr(obj, "children", []):
        print_tree(child, level+1)

# ancilliary data parsers

def parse_spacecraft_time(cds):
    epoch = np.datetime64("1958-01-01")
    return epoch + cds[...,0] * np.timedelta64(1, "D") \
                 + ((cds[...,1] << 16) + cds[...,2]) * np.timedelta64(1, "ms") \
                 + cds[...,3] * np.timedelta64(1, "us")

ANCILLARY_DATA_INFO = {
    "Time_Tag": {
        "parser": parse_spacecraft_time,
        "attrs": {
            "description": "spacecraft time (UTC)",
        },
        "name": "time",
        "dims": ("time",),
    },
    "Primary_Header": {
        "parser": None,
    },
    "Secondary_Header": {
        "parser": None,
    },
    "Flag_Byte": {
        "parser": None,
    },
    "Time_Conversion": {
        "parser": None,
    },
    "Position": {
        "parser": lambda p: p * 0.125,
        "attrs": {
            "units": "meter",
            "description": "Spacecraft Position (x, y, z) -- Estimated position of the spacecraft, expressed in Earth Centered Inertial frame (mean Equator and Equinox of J2000).",
        },
        "dims": ("time", "xyz"),
    },
    "Velocity": {
        "parser": lambda v: v * 244e-6,
        "attrs": {
            "units": "meter / second",
            "description": "Spacecraft Velocity (x, y, z) -- Estimated velocity of the spacecraft, expressed in Earth Centered Inertial frame (mean Equator and Equinox of J2000)."
        },
        "dims": ("time", "xyz"),
    },
    "Attitude_Angle": {
        "parser": lambda a: a * 1.0,
        "attrs": {
            "units": "arcsec",
            "description": "Attitude Angle (Roll, Pitch, Yaw) -- The estimated attitude of the spacecraft, expressed in the Orbital Reference frame.",
        },
        "dims": ("time", "rpy"),
    },
    "Attitude_Rate": {
        "parser": lambda r: r * 0.5,
        "attrs": {
            "units": "arcsec/sec",
            "description": "Attitude Rate (Roll, Pitch, Yaw) -- The estimated attitude rate of the spacecraft, expressed in the Orbital Reference frame.",
        },
        "dims": ("time", "rpy"),
    },
    "Magnetic_Coil": {
        "parser": lambda c: c * 15.6e-3,
        "attrs": {
            "units": "ampere",
            "description": "Magnetic Coil Current (x, y, z) -- Currents flowing in each of the magnetic torque coils used for Spacecraft momentum unloading.",
        },
        "dims": ("time", "xyz"),
    },
    "Solar_Array": {
        "parser": lambda a: a * 1.0,
        "attrs": {
            "units": "ampere",
            "description": "Current flowing from the Spacecraft solar array.",
        },
        "dims": ("time",),
    },
    "Solar_Position": {
        "parser": lambda p: p * 7.8e-3,
        "attrs": {
            "units": "1",
            "description": "Solar Position (x, y, z) -- Components of unit vector, expressed in the Spacecraft Reference frame, pointing in the direction of the Sun.",
        },
        "dims": ("time", "xyz"),
    },
    "Moon_Position": {
        "parser": lambda p: p * 7.8e-3,
        "attrs": {
            "units": "1",
            "description": "Moon Position (x, y, z) -- Components of the unit vector, expressed in the Spacecraft Reference frame, pointing in the direction of the Moon.",
        },
        "dims": ("time", "xyz"),
    },
}

def collect_ancillary(hdf):
    groups = [c for c in hdf.children if c.name == "Ancillary_Data"]
    vdata = [vd for c in groups for vd in c.children if vd.name == "Ancillary_Data"]
    fields = vdata[0].fields
    for vd in vdata:
        if vd.fields != fields:
            raise ValueError("inconsistend fields in Ancillary Data")
    data = [d for vd in vdata for d in vd.data]
    return xr.Dataset(
         {ANCILLARY_DATA_INFO[f].get("name", f): xr.DataArray(
                ANCILLARY_DATA_INFO[f]["parser"](np.array(d)),
                dims=ANCILLARY_DATA_INFO[f]["dims"],
                attrs=ANCILLARY_DATA_INFO[f].get("attrs", {}))
          for f, d in zip(fields, zip(*data))
          if f in ANCILLARY_DATA_INFO and ANCILLARY_DATA_INFO[f]["parser"] is not None}
        )

def sd2xrda(sd):
    return xr.DataArray(
        sd[:],
        dims=sd.dimnames,
        name=sd.name
    )

def collect_sensor(hdf, sensor):
    groups = [c for c in hdf.children if c.name == sensor and c.gclass == "1B"]
    assert len(groups) == 1, "unknown sensor name"
    group = groups[0]
    geoloc_fields = [f
                     for g1 in group.children if g1.name.endswith("_Swath")
                     for g2 in g1.children if g2.name == "Geolocation Fields"
                     for f in g2.children]
    data_fields = [f
                   for g1 in group.children if g1.name.endswith("_Swath")
                   for g2 in g1.children if g2.name == "Data Fields"
                   for f in g2.children]

    ds = xr.Dataset(
        {f.name: sd2xrda(f) for f in geoloc_fields + data_fields}
    )
    ds = ds.rename({"Longitude": "{}_Longitude".format(sensor),
                    "Latitude": "{}_Latitude".format(sensor)})
    return ds

def open_aster(filename):
    hdf = HDFReader(filename)
    print_tree(hdf)
    anc = collect_ancillary(hdf)
    vnir = collect_sensor(hdf, "VNIR")
    swir = collect_sensor(hdf, "SWIR")
    tir = collect_sensor(hdf, "TIR")
    #print(anc)
    #print(vnir)
    #print(swir)
    #print(tir)

    return xr.merge([anc, vnir, swir, tir])
