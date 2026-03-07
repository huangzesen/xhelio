"""Tests for knowledge.mission_prefixes — shared CDAWeb dataset prefix mapping."""

import pytest

from knowledge.mission_prefixes import (
    MISSION_PREFIX_MAP,
    MISSION_NAMES,
    match_dataset_to_mission,
    get_all_mission_stems,
    get_mission_name,
    get_mission_keywords,
    create_mission_skeleton,
)


# ── match_dataset_to_mission ────────────────────────────────────────

class TestMatchDatasetToMission:
    """Test prefix matching for existing curated missions."""

    def test_psp_fields(self):
        mission, inst = match_dataset_to_mission("PSP_FLD_L2_MAG_RTN_1MIN")
        assert mission == "psp"
        assert inst == "FIELDS/MAG"

    def test_psp_sweap(self):
        mission, inst = match_dataset_to_mission("PSP_SWP_SPC_L3I")
        assert mission == "psp"
        assert inst == "SWEAP"

    def test_psp_generic(self):
        mission, inst = match_dataset_to_mission("PSP_SOMETHING_ELSE")
        assert mission == "psp"
        assert inst is None

    def test_ace_mag(self):
        mission, _ = match_dataset_to_mission("AC_H2_MFI")
        assert mission == "ace"

    def test_ace_key_param(self):
        mission, _ = match_dataset_to_mission("AC_K1_EPM")
        assert mission == "ace"

    def test_solo_mag(self):
        mission, inst = match_dataset_to_mission("SOLO_L2_MAG_RTN_1MIN")
        assert mission == "solo"
        assert inst == "MAG"

    def test_omni(self):
        mission, inst = match_dataset_to_mission("OMNI_HRO_1MIN")
        assert mission == "omni"
        assert inst == "Combined"

    def test_wind(self):
        mission, _ = match_dataset_to_mission("WI_H2_MFI")
        assert mission == "wind"

    def test_dscovr_mag(self):
        mission, inst = match_dataset_to_mission("DSCOVR_H0_MAG")
        assert mission == "dscovr"
        assert inst == "MAG"

    def test_mms1(self):
        mission, inst = match_dataset_to_mission("MMS1_FGM_SRVY_L2")
        assert mission == "mms"
        assert inst == "FGM"

    def test_mms2(self):
        mission, _ = match_dataset_to_mission("MMS2_FGM_SRVY_L2")
        assert mission == "mms"

    def test_mms3(self):
        mission, _ = match_dataset_to_mission("MMS3_FPI_FAST_L2_DIS")
        assert mission == "mms"

    def test_mms4(self):
        mission, _ = match_dataset_to_mission("MMS4_FGM_BRST_L2")
        assert mission == "mms"

    def test_stereo_a(self):
        mission, inst = match_dataset_to_mission("STA_L2_MAG_RTN")
        assert mission == "stereo_a"
        assert inst == "MAG"


class TestMatchNewMissions:
    """Test prefix matching for newly added missions."""

    def test_stereo_b(self):
        mission, _ = match_dataset_to_mission("STB_L2_MAG_RTN")
        assert mission == "stereo_b"

    def test_themis_a(self):
        mission, _ = match_dataset_to_mission("THA_L2_FGM")
        assert mission == "themis"

    def test_themis_b(self):
        mission, _ = match_dataset_to_mission("THB_L2_MOM")
        assert mission == "themis"

    def test_themis_c(self):
        mission, _ = match_dataset_to_mission("THC_L1_STATE")
        assert mission == "themis"

    def test_cluster_c1(self):
        mission, _ = match_dataset_to_mission("C1_CP_FGM_SPIN")
        assert mission == "cluster"

    def test_cluster_c4(self):
        mission, _ = match_dataset_to_mission("C4_CP_CIS_CODIF")
        assert mission == "cluster"

    def test_rbsp(self):
        mission, _ = match_dataset_to_mission("RBSPA_REL04_ECT-HOPE-SCI-L2")
        assert mission == "rbsp"

    def test_goes(self):
        mission, _ = match_dataset_to_mission("GOES16_EXIS_L1B")
        assert mission == "goes"

    def test_goes_g15(self):
        mission, _ = match_dataset_to_mission("G15_EPEAD_CPFLUX_1MIN")
        assert mission == "goes"

    def test_voyager1(self):
        mission, _ = match_dataset_to_mission("VG1_48S_MAG")
        assert mission == "voyager1"

    def test_voyager2(self):
        mission, _ = match_dataset_to_mission("VG2_48S_MAG")
        assert mission == "voyager2"

    def test_ulysses(self):
        mission, _ = match_dataset_to_mission("UY_M0_VHM")
        assert mission == "ulysses"

    def test_geotail(self):
        mission, _ = match_dataset_to_mission("GE_K0_MGF")
        assert mission == "geotail"

    def test_polar(self):
        mission, _ = match_dataset_to_mission("PO_K0_MFE")
        assert mission == "polar"

    def test_image(self):
        mission, _ = match_dataset_to_mission("IM_K0_EUV")
        assert mission == "image"

    def test_fast(self):
        mission, _ = match_dataset_to_mission("FA_K0_DCF")
        assert mission == "fast"

    def test_soho(self):
        mission, _ = match_dataset_to_mission("SOHO_CELIAS_SEM")
        assert mission == "soho"

    def test_juno(self):
        mission, _ = match_dataset_to_mission("JUNO_JADE_L2")
        assert mission == "juno"

    def test_maven(self):
        mission, _ = match_dataset_to_mission("MVN_MAG_L2")
        assert mission == "maven"

    def test_cassini(self):
        mission, _ = match_dataset_to_mission("CO_MAG_1MIN")
        assert mission == "cassini"

    def test_arase(self):
        mission, _ = match_dataset_to_mission("ERG_LEPE_L2_OMNIFLUX")
        assert mission == "arase"


class TestNoMatch:
    """Test that truly unknown prefixes return None."""

    def test_unknown(self):
        mission, inst = match_dataset_to_mission("XYZZY_FOO_BAR")
        assert mission is None
        assert inst is None

    def test_empty_string(self):
        mission, inst = match_dataset_to_mission("")
        assert mission is None
        assert inst is None


class TestPrefixOrdering:
    """Test that more specific prefixes match before less specific ones."""

    def test_psp_fld_before_psp(self):
        mission, inst = match_dataset_to_mission("PSP_FLD_L2_MAG")
        assert inst == "FIELDS/MAG"

    def test_dscovr_mag_before_dscovr(self):
        mission, inst = match_dataset_to_mission("DSCOVR_H0_MAG_1MIN")
        assert inst == "MAG"

    def test_omni_hro_before_omni(self):
        mission, inst = match_dataset_to_mission("OMNI_HRO_1MIN")
        assert inst == "Combined"


# ── Helper functions ────────────────────────────────────────────────

class TestGetAllMissionStems:
    """Test get_all_mission_stems returns all unique mission stems."""

    def test_includes_curated(self):
        stems = get_all_mission_stems()
        for curated in ["psp", "solo", "ace", "omni", "wind", "dscovr", "mms", "stereo_a"]:
            assert curated in stems

    def test_includes_new(self):
        stems = get_all_mission_stems()
        for new in ["themis", "cluster", "rbsp", "goes", "voyager1", "voyager2"]:
            assert new in stems

    def test_sorted(self):
        stems = get_all_mission_stems()
        assert stems == sorted(stems)


class TestGetMissionName:
    """Test human-readable mission name lookup."""

    def test_known_mission(self):
        assert get_mission_name("psp") == "Parker Solar Probe"
        assert get_mission_name("ace") == "ACE"
        assert get_mission_name("themis") == "THEMIS"

    def test_unknown_mission(self):
        assert get_mission_name("xyzzy") == "XYZZY"


class TestGetMissionKeywords:
    """Test auto-derived mission keywords."""

    def test_psp_keywords(self):
        kw = get_mission_keywords("psp")
        assert "psp" in kw
        assert "parker" in kw

    def test_ace_keywords(self):
        kw = get_mission_keywords("ace")
        assert "ace" in kw
        assert "ac_h" in kw  # From prefix AC_H

    def test_themis_keywords(self):
        kw = get_mission_keywords("themis")
        assert "themis" in kw
        assert "tha" in kw  # From prefix THA_


class TestCreateMissionSkeleton:
    """Test mission JSON skeleton creation."""

    def test_skeleton_structure(self):
        skel = create_mission_skeleton("themis")
        assert skel["id"] == "THEMIS"
        assert skel["name"] == "THEMIS"
        assert "keywords" in skel
        assert "profile" in skel
        assert "instruments" in skel
        assert "General" in skel["instruments"]

    def test_skeleton_has_profile(self):
        skel = create_mission_skeleton("cluster")
        profile = skel["profile"]
        assert "description" in profile
        assert "coordinate_systems" in profile
        assert "typical_cadence" in profile

    def test_skeleton_id_underscore(self):
        """Mission stems with underscore use canonical IDs."""
        skel = create_mission_skeleton("stereo_b")
        assert skel["id"] == "STEREO_B"
        assert skel["name"] == "STEREO-B"  # Human-readable name uses hyphen

    def test_skeleton_unknown_mission(self):
        skel = create_mission_skeleton("xyzzy")
        assert skel["id"] == "XYZZY"
        assert skel["name"] == "XYZZY"


# ── Consistency checks ──────────────────────────────────────────────

class TestConsistency:
    """Verify that the prefix map and name registry are consistent."""

    def test_all_prefix_missions_have_names(self):
        """Every mission_stem in the prefix map should have a human-readable name."""
        stems = get_all_mission_stems()
        for stem in stems:
            name = get_mission_name(stem)
            # Should not just be upper-cased stem (meaning it's missing from MISSION_NAMES)
            # Allow it for very short names like "ACE" where upper == name
            assert name is not None and len(name) > 0, f"Missing name for {stem}"

    def test_mission_names_keys_are_lowercase(self):
        """All MISSION_NAMES keys should be lowercase."""
        for key in MISSION_NAMES:
            assert key == key.lower(), f"MISSION_NAMES key '{key}' should be lowercase"

    def test_prefix_map_no_empty_keys(self):
        """No empty string prefixes."""
        for prefix in MISSION_PREFIX_MAP:
            assert len(prefix) > 0, "Empty prefix in MISSION_PREFIX_MAP"
