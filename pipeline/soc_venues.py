"""Canonical sociology-venue list, shared across the pipeline.

Single source of truth for "which journals count as sociology" — used by
`ingest_journal_authors.py` (the journal sweep) and `build_discipline_signal.py`
(the per-person discipline / US-sociology-orientation signal).

Each entry is (issn_l, name, us) where `us` marks a journal as anchored in
*American* sociology. The `us` flag is a deliberate editorial call (point of the
signal is "tied to US sociology"), so it's easy to eyeball and tweak here:
British / European / other-anchored venues are `us=False`, everything else True.

`journal_issn_l` in works.parquet is the linking ISSN. One venue (Social
Movement Studies) carries a different ISSN-L in OpenAlex than its print ISSN, so
NAME_ALIASES provides an exact-name fallback for the handful that need it.
"""

# (issn_l, journal name, is_us_anchored)
SOC_VENUES: list[tuple[str, str, bool]] = [
    # Generalist sociology
    ("0003-1224", "American Sociological Review", True),
    ("0002-9602", "American Journal of Sociology", True),
    ("0037-7732", "Social Forces", True),
    ("0037-7791", "Social Problems", True),
    ("2378-0231", "Socius", True),
    ("2330-6696", "Sociological Science", True),
    ("0007-1315", "British Journal of Sociology", False),
    ("0266-7215", "European Sociological Review", False),
    ("0360-0572", "Annual Review of Sociology", True),
    ("0884-8971", "Sociological Forum", True),
    ("0735-2751", "Sociological Theory", True),
    ("0731-1214", "Sociological Perspectives", True),
    ("2329-4965", "Social Currents", True),
    ("1536-5042", "Contexts", True),
    ("0002-7642", "American Behavioral Scientist", True),
    ("2377-8253", "RSF: The Russell Sage Foundation Journal of the Social Sciences", True),
    # Population / demography / migration
    ("0070-3370", "Demography", True),
    ("0098-7921", "Population and Development Review", True),
    ("1435-9871", "Demographic Research", False),
    ("0197-9183", "International Migration Review", True),
    ("1369-183X", "Journal of Ethnic and Migration Studies", False),
    # Stratification / education / work / economy
    ("0038-0407", "Sociology of Education", True),
    ("0730-8884", "Work and Occupations", True),
    ("0891-2432", "Gender & Society", True),
    ("1475-1461", "Socio-Economic Review", False),
    # Family / health
    ("0022-2445", "Journal of Marriage and Family", True),
    ("0022-1465", "Journal of Health and Social Behavior", True),
    ("0192-513X", "Journal of Family Issues", True),
    ("2352-8273", "SSM - Population Health", False),
    # Crime / law
    ("0011-1384", "Criminology", True),
    ("0023-9216", "Law & Society Review", True),
    # Movements / politics
    ("1086-671X", "Mobilization", True),
    ("1474-2837", "Social Movement Studies", False),
    # Methods
    ("0049-1241", "Sociological Methods & Research", True),
    ("0081-1750", "Sociological Methodology", True),
    # Race / ethnicity
    ("1867-1748", "Race and Social Problems", True),
    ("2332-6492", "Sociology of Race and Ethnicity", True),
    ("0141-9870", "Ethnic and Racial Studies", False),
    # Religion
    ("0021-8294", "Journal for the Scientific Study of Religion", True),
]

# Back-compat alias for ingest_journal_authors.py (ISSN, name) tuples.
DEFAULT_VENUES: list[tuple[str, str]] = [(i, n) for i, n, _ in SOC_VENUES]

SOC_ISSNS: frozenset[str] = frozenset(i for i, _, _ in SOC_VENUES)
US_SOC_ISSNS: frozenset[str] = frozenset(i for i, _, us in SOC_VENUES if us)

# Exact journal-name fallback for venues whose OpenAlex ISSN-L differs from the
# print ISSN above (so an ISSN-only join would miss them). Maps name → issn_l.
NAME_ALIASES: dict[str, str] = {
    "Social Movement Studies": "1474-2837",
}
