#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:35:46 2023

@author: e210
"""
import re

def regex_search_label(regexes, label):
    if any(regex.search(label) for regex in regexes):
            return True
    else:
            return False
        
def is_t1(ds):
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")
    regexes = [
        re.compile('t1', re.IGNORECASE),
        re.compile('t1w', re.IGNORECASE),
        re.compile('(?=.*3d anat)(?![inplane])', re.IGNORECASE),
        re.compile('(?=.*3d)(?=.*bravo)(?![inplane])', re.IGNORECASE),
        re.compile('spgr', re.IGNORECASE),
        re.compile('tfl', re.IGNORECASE),
        re.compile('mprage', re.IGNORECASE),
        re.compile('(?=.*mm)(?=.*iso)', re.IGNORECASE),
        re.compile('(?=.*mp)(?=.*rage)', re.IGNORECASE)
    ]

    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)

def is_t2(ds):
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")
    regexes = [
        re.compile('t2', re.IGNORECASE),
        re.compile('t2w', re.IGNORECASE),

    ]

    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)
def is_mrcp(ds):
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")
    regexes = [
         re.compile(r'mrcp', re.IGNORECASE),
         re.compile(r'cholangiopancreatography', re.IGNORECASE),
         re.compile(r'pancreatography', re.IGNORECASE)
     ]
    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)


#### second level ####


#### second level ####
def with_contrast(ds):
    # Check for direct contrast indicators
    contrast = getattr(ds, "ContrastBolusAgent", "")
    contrast_route = getattr(ds, "ContrastBolusRoute", "")

    if (contrast and contrast != "LO") or (contrast_route and contrast_route != "LO"):
        return True

    # Prepare regexes for SeriesDescription and ProtocolName
    regexes = [
        re.compile('KM', re.IGNORECASE),
        re.compile('POST.*CONTRAST', re.IGNORECASE),
        re.compile('CECT', re.IGNORECASE),
        re.compile('GAD', re.IGNORECASE),
        re.compile('GD', re.IGNORECASE)
    ]

    # Check SeriesDescription and ProtocolName
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")

    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)
    
def with_fat_saturation(ds):
    # Check for direct contrast indicators
    scan_option = getattr(ds, "ScanOptions", "")


    if ("FS" in scan_option) or ("SFS" in scan_option):
        return True

    # Check SeriesDescription and ProtocolName
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")
    regexes = [
        re.compile('fs', re.IGNORECASE)
    ]
    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)

###### thrid levl #####


def is_gre_dixon_w(ds):
    regexes = [
        re.compile(r'(Dixon.*VIBE)|(VIBE.*Dixon)', re.IGNORECASE),
        re.compile(r'LAVA-Flex', re.IGNORECASE),
        re.compile(r'mDixon', re.IGNORECASE),
        re.compile(r'WFS', re.IGNORECASE)
    ]
    # Check SeriesDescription and ProtocolName
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")

    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)

def is_gre_tfisp(ds):
    regexes = [
        re.compile(r'true fisp', re.IGNORECASE),
        re.compile(r'truefisp', re.IGNORECASE),
        re.compile(r'fiesta', re.IGNORECASE),
        re.compile(r'cosmic', re.IGNORECASE),
        re.compile(r'balanced ffe', re.IGNORECASE),
        re.compile(r'true ssfp', re.IGNORECASE),
        re.compile(r'balanced sarge', re.IGNORECASE),
        re.compile(r'basg', re.IGNORECASE),
        re.compile(r'trufi', re.IGNORECASE)
    ]
    # Check SeriesDescription and ProtocolName
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")

    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)

def is_gre_tfispde(ds):
    regexes = [
        re.compile(r'CISS', re.IGNORECASE),
        re.compile(r'fiesta-c', re.IGNORECASE),

        re.compile(r'phase balanced sarge', re.IGNORECASE),
        re.compile(r'bpsg', re.IGNORECASE)
    ]
    # Check SeriesDescription and ProtocolName
    sd = getattr(ds, "SeriesDescription", "")
    pn = getattr(ds, "ProtocolName", "")

    return regex_search_label(regexes, sd) or regex_search_label(regexes, pn)
def generate_label(ds):
    levels = {
        # Level 1: Primary imaging type
        1: [(is_t1, "T1"), (is_t2, "T2"), (is_mrcp, "MRCP")],

        # Level 2: Enhancements like contrast and fat saturation
        2: [(with_contrast, "C"), (with_fat_saturation, "FS")],

        # Level 3: Additional classifications or checks can be added here
        3:[(is_gre_tfisp, "TFISP-GRE"), (is_gre_dixon_w, "DIXW-GRE"),(is_gre_tfispde, "TFISPDE-GRE")]
    }

    label_parts = []

    for level in sorted(levels.keys()):
        level_labels = [check_label for check_func, check_label in levels[level] if check_func(ds)]
        if level_labels:
            label_parts.append('-'.join(level_labels))

    return '_'.join(label_parts) if label_parts else "other"


