#!/usr/bin/env python

"""
Read the row-matched WISEsize tables.

The tables are tied to one canonical SGA parent sample.  Every derived table
should have the same number of rows and the same SGAID ordering as
``wisesize_sga_v1.fits``.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.units import UnitsWarning


class wstables:
    """Class containing the row-matched WISEsize tables."""

    table_names = {
        "parent": "wisesize_sga_v1.fits",
        "cigale": "wisesize_cigale_v1.fits",
        "vfs": "wisesize_vfs_v1.fits",
        "virgowise": "wisesize_virgowise.fits",
        "galfitr": "wisesize_galfit_r_v1.fits",
        "galfitW1fixBA": "wisesize_galfit_W1_fixBA_v1.fits",
        "galfitW3fixBA": "wisesize_galfit_W3_fixBA_v1.fits",
        "oldparent": "wisesize_parent_v0_matched_v1.fits",
        "m200": "wisesize_m200_v1.fits",
        "nedlvs": "wisesize_nedlvs_v1.fits",
    }

    def __init__(self, tabledir=None):
        """Class containing row-matched tables for the WISEsize project."""
        if tabledir is None:
            tabledir = "/Users/rfinn/research/WISEsize/tables/"
        self.tabledir = Path(os.path.expanduser(tabledir))
        self.sgaid = None

    def filename(self, key):
        """Return the full path to a table by key."""
        return self.tabledir / self.table_names[key]

    def read_table(self, key):
        """Read one table by key."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            return Table.read(self.filename(key))

    def validate_table(self, tab, name):
        """Verify that a table is row-matched to the parent table."""
        if self.sgaid is None:
            raise ValueError("Read the parent table before validating derived tables.")

        if len(tab) != len(self.sgaid):
            raise ValueError(f"{name}: row count does not match parent table.")

        if "SGAID" not in tab.colnames:
            raise ValueError(f"{name}: missing SGAID column.")

        if not np.array_equal(np.asarray(tab["SGAID"]), self.sgaid):
            raise ValueError(f"{name}: SGAID order does not match parent table.")

    def read_all(self):
        """Read all row-matched WISEsize tables."""
        self.read_parent()
        self.read_cigale()
        self.read_vfs()
        self.read_virgowise()
        self.read_galfit()
        self.read_oldparent()
        self.read_m200()
        self.read_nedlvs()

    def read_parent(self):
        """Read the canonical SGA parent table; store as self.parent."""
        self.parent = self.read_table("parent")
        self.sgaid = np.asarray(self.parent["SGAID"])

    def read_cigale(self):
        """Read CIGALE results matched to the parent; store as self.cigale."""
        self.cigale = self.read_table("cigale")
        self.validate_table(self.cigale, "cigale")

    def read_vfs(self):
        """Read Virgo Filaments Survey table matched to the parent; store as self.vfs."""
        self.vfs = self.read_table("vfs")
        self.validate_table(self.vfs, "vfs")

    def read_virgowise(self):
        """Read VirgoWISE data matched to the parent; store as self.virgowise."""
        self.virgowise = self.read_table("virgowise")
        self.validate_table(self.virgowise, "virgowise")

    def read_galfit(self):
        """Read GALFIT tables; store as self.galfitr, self.galfitW1fixBA, self.galfitW3fixBA."""
        self.galfitr = self.read_table("galfitr")
        self.validate_table(self.galfitr, "galfitr")

        self.galfitW1fixBA = self.read_table("galfitW1fixBA")
        self.validate_table(self.galfitW1fixBA, "galfitW1fixBA")

        self.galfitW3fixBA = self.read_table("galfitW3fixBA")
        self.validate_table(self.galfitW3fixBA, "galfitW3fixBA")

    def read_oldparent(self):
        """Read the old WISEsize parent table matched to the parent; store as self.oldparent."""
        self.oldparent = self.read_table("oldparent")
        self.validate_table(self.oldparent, "oldparent")

    def read_m200(self):
        """Read M200 table matched to the parent; store as self.m200."""
        self.m200 = self.read_table("m200")
        self.validate_table(self.m200, "m200")

    def read_nedlvs(self):
        """Read NED-LVS table matched to the parent; store as self.nedlvs."""
        self.nedlvs = self.read_table("nedlvs")
        self.validate_table(self.nedlvs, "nedlvs")

    def match_summary(self):
        """Return a table summarizing MATCH_FLAG counts for loaded derived tables."""
        names = []
        nrows = []
        nmatches = []
        for attr in self.table_names:
            if not hasattr(self, attr):
                continue
            tab = getattr(self, attr)
            names.append(attr)
            nrows.append(len(tab))
            if "MATCH_FLAG" in tab.colnames:
                nmatches.append(int(np.count_nonzero(tab["MATCH_FLAG"])))
            else:
                nmatches.append(len(tab))
        return Table([names, nrows, nmatches], names=["table", "nrows", "nmatches"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read row-matched WISEsize tables.")
    parser.add_argument(
        "--tabledir",
        default="/Users/rfinn/research/WISEsize/tables/",
        help="Directory where WISEsize tables are stored.",
    )
    args = parser.parse_args()

    w = wstables(args.tabledir)
    w.read_all()
    print("table directory = ", w.tabledir)
    print(w.match_summary())
