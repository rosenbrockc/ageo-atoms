#!/usr/bin/env python3
"""Bulk scholarly reference audit for all reviewed atoms.

This script rebuilds the global reference registry and per-directory
``references.json`` files from deterministic rules:

- upstream repository citations for mapped third_party implementations
- package/software papers for major external libraries
- method papers/books for atoms whose names map cleanly to canonical work

It writes the multi-atom ``references.json`` schema:

{
  "schema_version": "1.1",
  "atoms": {
    "<atom_id>": {
      "references": [{"ref_id": "...", "match_metadata": {...}}],
      "auto_attribution_runs": []
    }
  }
}
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "hyperparams" / "manifest.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
ATOM_MANIFEST_PATH = ROOT / "scripts" / "atom_manifest.yml"


def paper_ref(
    ref_id: str,
    *,
    title: str,
    authors: list[str],
    year: int,
    venue: str,
    doi: str | None = None,
    url: str | None = None,
    ref_type: str = "paper",
    notes: str,
) -> dict:
    ref = {
        "ref_id": ref_id,
        "type": ref_type,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "match_metadata": {
            "similarity_score": None,
            "match_type": "manual",
            "matched_nodes": [],
            "confidence": "high",
            "notes": notes,
        },
    }
    if doi:
        ref["doi"] = doi
    if url:
        ref["url"] = url
    return ref


def repo_ref(ref_id: str, repo_name: str, repo_url: str) -> dict:
    clean_url = repo_url[:-4] if repo_url.endswith(".git") else repo_url
    return {
        "ref_id": ref_id,
        "type": "repository",
        "title": repo_name,
        "authors": [],
        "year": None,
        "venue": "GitHub",
        "url": clean_url,
        "match_metadata": {
            "similarity_score": None,
            "match_type": "manual",
            "matched_nodes": [],
            "confidence": "high",
            "notes": "Repository citation for the upstream implementation used by this atom.",
        },
    }


MANUAL_REFS: dict[str, dict] = {
    "numpy2020": paper_ref(
        "numpy2020",
        title="Array programming with NumPy",
        authors=[
            "Charles R. Harris",
            "K. Jarrod Millman",
            "Stéfan J. van der Walt",
            "Ralf Gommers",
            "Pauli Virtanen",
            "David Cournapeau",
            "Eric Wieser",
            "Julian Taylor",
            "Sebastian Berg",
            "Nathaniel J. Smith",
            "Robert Kern",
            "Matti Picus",
            "Stephan Hoyer",
            "Marten H. van Kerkwijk",
            "Matthew Brett",
            "Allan Haldane",
            "Jaime Fernández del Río",
            "Mark Wiebe",
            "Pearu Peterson",
            "Pierre Gérard-Marchant",
            "Kevin Sheppard",
            "Tyler Reddy",
            "Warren Weckesser",
            "Hameer Abbasi",
            "Christoph Gohlke",
            "Travis E. Oliphant",
        ],
        year=2020,
        venue="Nature",
        doi="10.1038/s41586-020-2649-2",
        url="https://doi.org/10.1038/s41586-020-2649-2",
        notes="Canonical NumPy software paper.",
    ),
    "scipy2020": paper_ref(
        "scipy2020",
        title="SciPy 1.0: fundamental algorithms for scientific computing in Python",
        authors=[
            "Pauli Virtanen",
            "Ralf Gommers",
            "Travis E. Oliphant",
            "Matt Haberland",
            "Tyler Reddy",
            "David Cournapeau",
            "Evgeni Burovski",
            "Pearu Peterson",
            "Warren Weckesser",
            "Jonathan Bright",
            "Stéfan J. van der Walt",
            "Matthew Brett",
            "Joshua Wilson",
            "K. Jarrod Millman",
            "Nikolay Mayorov",
            "Andrew R. J. Nelson",
            "Eric Jones",
            "Robert Kern",
            "Eric Larson",
            "C J Carey",
            "İlhan Polat",
            "Yu Feng",
            "Eric W. Moore",
            "Jake VanderPlas",
            "Denis Laxalde",
            "Josef Perktold",
            "Robert Cimrman",
            "Ian Henriksen",
            "E. A. Quintero",
            "Charles R. Harris",
            "Anne M. Archibald",
            "Antony H. Ribeiro",
            "Fabian Pedregosa",
            "Paul van Mulbregt",
            "SciPy 1.0 Contributors",
        ],
        year=2020,
        venue="Nature Methods",
        doi="10.1038/s41592-019-0686-2",
        url="https://doi.org/10.1038/s41592-019-0686-2",
        notes="Canonical SciPy software paper.",
    ),
    "alphafold2021": paper_ref(
        "alphafold2021",
        title="Highly accurate protein structure prediction with AlphaFold",
        authors=[
            "John Jumper",
            "Richard Evans",
            "Alexander Pritzel",
            "Tim Green",
            "Michael Figurnov",
            "Olaf Ronneberger",
            "Kathryn Tunyasuvunakool",
            "Russ Bates",
            "Augustin Žídek",
            "Anna Potapenko",
            "Alex Bridgland",
            "Clemens Meyer",
            "Simon A. A. Kohl",
            "Andrew J. Ballard",
            "Andrew Cowie",
            "Bernardino Romera-Paredes",
            "Stanislav Nikolov",
            "Rishub Jain",
            "Jonas Adler",
            "Trevor Back",
            "Stig Petersen",
            "David Reiman",
            "Ellen Clancy",
            "Michal Zielinski",
            "Martin Steinegger",
            "Michal Pacholska",
            "Tamas Berghammer",
            "David Silver",
            "Oriol Vinyals",
            "Andrew W. Senior",
            "Koray Kavukcuoglu",
            "Pushmeet Kohli",
            "Demis Hassabis",
        ],
        year=2021,
        venue="Nature",
        doi="10.1038/s41586-021-03819-2",
        url="https://doi.org/10.1038/s41586-021-03819-2",
        notes="Canonical AlphaFold2 paper.",
    ),
    "camurri2020pronto": paper_ref(
        "camurri2020pronto",
        title="Pronto: A Multi-Sensor State Estimator for Legged Robots in Real-World Scenarios",
        authors=["Marco Camurri", "Milad Ramezani", "Simona Nobili", "Maurice Fallon"],
        year=2020,
        venue="Frontiers in Robotics and AI",
        doi="10.3389/frobt.2020.00068",
        url="https://doi.org/10.3389/frobt.2020.00068",
        notes="Primary Pronto software paper cited by the upstream repository.",
    ),
    "advancedhmc2020": paper_ref(
        "advancedhmc2020",
        title="AdvancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms",
        authors=["Kai Xu", "Tamas Papp", "Bálint Aradi", "Michael Innes", "Alan Edelman"],
        year=2020,
        venue="Proceedings of Machine Learning Research",
        url="https://proceedings.mlr.press/v118/xu20a.html",
        notes="Upstream package paper for AdvancedHMC.jl.",
    ),
    "neurokit2021": paper_ref(
        "neurokit2021",
        title="NeuroKit2: A Python toolbox for neurophysiological signal processing",
        authors=[
            "Dominique Makowski",
            "Tam Pham",
            "Zen J. Lau",
            "Jan C. Brammer",
            "François Lespinasse",
            "Hung Pham",
            "Christopher Schölzel",
            "S. H. Annabel Chen",
        ],
        year=2021,
        venue="Behavior Research Methods",
        doi="10.3758/s13428-020-01516-y",
        url="https://doi.org/10.3758/s13428-020-01516-y",
        notes="Preferred NeuroKit2 citation from upstream CITATION.cff.",
    ),
    "skyfield2019ascl": paper_ref(
        "skyfield2019ascl",
        title="Skyfield: Generate high precision research-grade positions for stars, planets, moons, and Earth satellites",
        authors=["Brandon Rhodes"],
        year=2019,
        venue="Astrophysics Source Code Library",
        url="https://ui.adsabs.harvard.edu/abs/2019ascl.soft07024R",
        ref_type="standard",
        notes="Preferred Skyfield citation from upstream CITATION.cff.",
    ),
    "astroflow2025": paper_ref(
        "astroflow2025",
        title="ASTROFLOW: A Real-Time End-to-End Pipeline for Radio Single-Pulse Searches",
        authors=[],
        year=2025,
        venue="arXiv",
        url="https://arxiv.org/abs/2511.02328",
        notes="Citation requested by the upstream AstroFlow repository.",
    ),
    "moleculardocking2025": paper_ref(
        "moleculardocking2025",
        title="A Scalable Heuristic for Molecular Docking on Neutral-Atom Quantum Processors",
        authors=["Mathieu Garrigues", "Victor Onofre", "Wesley Coelho", "S. Acheche"],
        year=2025,
        venue="arXiv",
        url="https://arxiv.org/abs/2508.18147",
        notes="Primary paper cited by the upstream Molecular-Docking repository.",
    ),
    "mint2026": paper_ref(
        "mint2026",
        title="Learning the language of protein-protein interactions",
        authors=[],
        year=2026,
        venue="Nature Communications",
        doi="10.1038/s41467-026-56272-5",
        url="https://doi.org/10.1038/s41467-026-56272-5",
        notes="Primary paper cited by the upstream mint repository.",
    ),
    "feli2023pipeline": paper_ref(
        "feli2023pipeline",
        title="End-to-End PPG Processing Pipeline for Wearables: From Quality Assessment and Motion Artifacts Removal to HR/HRV Feature Extraction",
        authors=[
            "Mohammad Feli",
            "Kianoosh Kazemi",
            "Iman Azimi",
            "Yuning Wang",
            "Amir Rahmani",
            "Pasi Liljeberg",
        ],
        year=2023,
        venue="2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)",
        url="https://ieeexplore.ieee.org/document/10423165",
        notes="Primary pipeline paper cited by the upstream E2E-PPG repository.",
    ),
    "feli2023sqa": paper_ref(
        "feli2023sqa",
        title="An Energy-Efficient Semi-Supervised Approach for On-Device Photoplethysmogram Signal Quality Assessment",
        authors=[
            "Mohammad Feli",
            "Iman Azimi",
            "Arman Anzanpour",
            "Amir M. Rahmani",
            "Pasi Liljeberg",
        ],
        year=2023,
        venue="Smart Health",
        url="https://www.sciencedirect.com/science/article/pii/S2352648323000201",
        notes="E2E-PPG signal quality assessment paper cited by the upstream repository.",
    ),
    "wang2022ppg": paper_ref(
        "wang2022ppg",
        title="PPG Signal Reconstruction Using Deep Convolutional Generative Adversarial Network",
        authors=[
            "Yuning Wang",
            "Iman Azimi",
            "Kianoosh Kazemi",
            "Amir M. Rahmani",
            "Pasi Liljeberg",
        ],
        year=2022,
        venue="2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)",
        url="https://ieeexplore.ieee.org/document/9870821",
        notes="E2E-PPG reconstruction paper cited by the upstream repository.",
    ),
    "kazemi2022ppg": paper_ref(
        "kazemi2022ppg",
        title="Robust PPG Peak Detection Using Dilated Convolutional Neural Networks",
        authors=[
            "Kianoosh Kazemi",
            "Juho Laitala",
            "Iman Azimi",
            "Pasi Liljeberg",
            "Amir M. Rahmani",
        ],
        year=2022,
        venue="Sensors",
        doi="10.3390/s22166054",
        url="https://doi.org/10.3390/s22166054",
        notes="E2E-PPG peak detection paper cited by the upstream repository.",
    ),
    "hamilton1986": paper_ref(
        "hamilton1986",
        title="Quantitative Investigation of QRS Detection Rules Using the MIT/BIH Arrhythmia Database",
        authors=["P. S. Hamilton", "W. J. Tompkins"],
        year=1986,
        venue="IEEE Transactions on Biomedical Engineering",
        doi="10.1109/TBME.1986.325695",
        url="https://doi.org/10.1109/TBME.1986.325695",
        notes="Foundational Hamilton-Tompkins ECG QRS detector paper.",
    ),
    "engzee1979": paper_ref(
        "engzee1979",
        title="A Single Scan Algorithm for QRS-Detection and Feature Extraction",
        authors=["W. Engelse", "C. Zeelenberg"],
        year=1979,
        venue="Computers in Cardiology",
        doi="10.1109/TBME.1979.326338",
        url="https://doi.org/10.1109/TBME.1979.326338",
        notes="Original Engelse-Zeelenberg ECG detector paper.",
    ),
    "christov2004": paper_ref(
        "christov2004",
        title="Real time electrocardiogram QRS detection using combined adaptive threshold",
        authors=["Ivaylo I. Christov"],
        year=2004,
        venue="BioMedical Engineering OnLine",
        doi="10.1186/1475-925X-3-28",
        url="https://doi.org/10.1186/1475-925X-3-28",
        notes="Exact citation named in the upstream BioSPPy Christov implementation.",
    ),
    "zhao2018ecgsqi": paper_ref(
        "zhao2018ecgsqi",
        title="SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation",
        authors=["Zhao Zhao", "Yuanting Zhang"],
        year=2018,
        venue="Frontiers in Physiology",
        doi="10.3389/fphys.2018.00727",
        url="https://doi.org/10.3389/fphys.2018.00727",
        notes="Exact SQI paper named in the upstream BioSPPy ZZ2018 implementation.",
    ),
    "zong2003abp": paper_ref(
        "zong2003abp",
        title="An Open-source Algorithm to Detect Onset of Arterial Blood Pressure Pulses",
        authors=["W. Zong", "T. Heldt", "G. B. Moody", "R. G. Mark"],
        year=2003,
        venue="Computers in Cardiology",
        url="https://physionet.org/content/cardiac-output/1.0.0/papers/w-zong-1.pdf",
        notes="Exact citation named in the upstream BioSPPy ABP onset implementation.",
    ),
    "elgendi2013ppg": paper_ref(
        "elgendi2013ppg",
        title="Systolic Peak Detection in Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions",
        authors=["Mohamed Elgendi", "Ian Norton", "Mark Brearley", "Derek Abbott", "Dale Schuurmans"],
        year=2013,
        venue="PLOS ONE",
        doi="10.1371/journal.pone.0076585",
        url="https://doi.org/10.1371/journal.pone.0076585",
        notes="Exact citation named in the upstream BioSPPy PPG onset implementation.",
    ),
    "kavsaoglu2016ppg": paper_ref(
        "kavsaoglu2016ppg",
        title="An innovative peak detection algorithm for photoplethysmography signals: An adaptive segmentation method",
        authors=["Ahmet R. Kavsaoğlu", "Kemal Polat", "Mehmet R. Bozkurt"],
        year=2016,
        venue="Turkish Journal of Electrical Engineering and Computer Sciences",
        doi="10.3906/elk-1310-177",
        url="https://doi.org/10.3906/elk-1310-177",
        notes="Exact citation named in the upstream BioSPPy PPG onset implementation.",
    ),
    "gutierrezrivas2015": paper_ref(
        "gutierrezrivas2015",
        title="Novel Real-Time Low-Complexity QRS Complex Detector Based on Adaptive Thresholding",
        authors=["R. Gutiérrez-Rivas", "J. J. García", "W. P. Marnane", "A. Hernández"],
        year=2015,
        venue="Journal of Medical Systems",
        url="https://link.springer.com/article/10.1007/s10916-015-0240-2",
        notes="Primary adaptive-threshold reference named in the upstream BioSPPy ASI implementation.",
    ),
    "sadhukhan2012": paper_ref(
        "sadhukhan2012",
        title="R-Peak Detection Algorithm for ECG using Double Difference and RR Interval Processing",
        authors=["Debargha Sadhukhan"],
        year=2012,
        venue="Procedia Technology",
        url="https://www.sciencedirect.com/science/article/pii/S2212017312005316",
        notes="Secondary reference named in the upstream BioSPPy ASI implementation.",
    ),
    "bonato1998": paper_ref(
        "bonato1998",
        title="A statistical method for the measurement of muscle activation intervals from surface myoelectric signal during gait",
        authors=["Paolo Bonato", "Tommaso D’Alessio", "Marco Knaflitz"],
        year=1998,
        venue="IEEE Transactions on Biomedical Engineering",
        url="https://pubmed.ncbi.nlm.nih.gov/9522545/",
        notes="Exact citation named in the upstream BioSPPy EMG Bonato implementation.",
    ),
    "abbink1998": paper_ref(
        "abbink1998",
        title="Detection of onset and termination of muscle activity in surface electromyograms",
        authors=["J. H. Abbink", "A. van der Bilt", "H. W. van der Glas"],
        year=1998,
        venue="Journal of Oral Rehabilitation",
        url="https://pubmed.ncbi.nlm.nih.gov/9639159/",
        notes="Exact citation named in the upstream BioSPPy EMG Abbink implementation.",
    ),
    "solnik2010": paper_ref(
        "solnik2010",
        title="Teager-Kaiser energy operator signal conditioning improves EMG onset detection",
        authors=["Stanley Solnik", "Peter Rider", "Kevin Steinweg", "Paul DeVita", "Tibor Hortobágyi"],
        year=2010,
        venue="European Journal of Applied Physiology",
        url="https://pubmed.ncbi.nlm.nih.gov/20238101/",
        notes="Exact citation named in the upstream BioSPPy EMG Solnik implementation.",
    ),
    "kalman1960": paper_ref(
        "kalman1960",
        title="A New Approach to Linear Filtering and Prediction Problems",
        authors=["R. E. Kalman"],
        year=1960,
        venue="Transactions of the ASME Journal of Basic Engineering",
        doi="10.1115/1.3662552",
        url="https://doi.org/10.1115/1.3662552",
        notes="Foundational Kalman filter paper.",
    ),
    "almgren2000": paper_ref(
        "almgren2000",
        title="Optimal Execution of Portfolio Transactions",
        authors=["Robert Almgren", "Neil Chriss"],
        year=2000,
        venue="Journal of Risk",
        url="https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf",
        notes="Canonical Almgren-Chriss execution paper.",
    ),
    "avellaneda2008": paper_ref(
        "avellaneda2008",
        title="High-frequency trading in a limit order book",
        authors=["Marco Avellaneda", "Sasha Stoikov"],
        year=2008,
        venue="Quantitative Finance",
        doi="10.1080/14697680701381228",
        url="https://doi.org/10.1080/14697680701381228",
        notes="Canonical Avellaneda-Stoikov market-making paper.",
    ),
    "cont2014ofi": paper_ref(
        "cont2014ofi",
        title="The Price Impact of Order Book Events",
        authors=["Rama Cont", "Arseniy Kukanov", "Sasha Stoikov"],
        year=2014,
        venue="Journal of Financial Econometrics",
        doi="10.1093/jjfinec/nbt003",
        url="https://doi.org/10.1093/jjfinec/nbt003",
        notes="Canonical order flow imbalance and order-book event reference.",
    ),
    "heston1993": paper_ref(
        "heston1993",
        title="A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options",
        authors=["Steven L. Heston"],
        year=1993,
        venue="The Review of Financial Studies",
        doi="10.1093/rfs/6.2.327",
        url="https://doi.org/10.1093/rfs/6.2.327",
        notes="Canonical Heston stochastic volatility paper.",
    ),
    "nelsen2006copulas": paper_ref(
        "nelsen2006copulas",
        title="An Introduction to Copulas",
        authors=["Roger B. Nelsen"],
        year=2006,
        venue="Springer",
        url="https://link.springer.com/book/10.1007/0-387-28678-0",
        ref_type="book",
        notes="Standard reference for copula modeling.",
    ),
    "coles2001evt": paper_ref(
        "coles2001evt",
        title="An Introduction to Statistical Modeling of Extreme Values",
        authors=["Stuart Coles"],
        year=2001,
        venue="Springer",
        url="https://link.springer.com/book/10.1007/978-1-4471-3675-0",
        ref_type="book",
        notes="Standard EVT reference used for GPD tail fitting attribution.",
    ),
    "hawkes1971": paper_ref(
        "hawkes1971",
        title="Spectra of some self-exciting and mutually exciting point processes",
        authors=["Alan G. Hawkes"],
        year=1971,
        venue="Biometrika",
        doi="10.1093/biomet/58.1.83",
        url="https://doi.org/10.1093/biomet/58.1.83",
        notes="Foundational Hawkes process paper.",
    ),
    "easley1996pin": paper_ref(
        "easley1996pin",
        title="Liquidity, Information, and Infrequently Traded Stocks",
        authors=["David Easley", "Nicholas M. Kiefer", "Maureen O’Hara", "Joseph B. Paperman"],
        year=1996,
        venue="The Journal of Finance",
        doi="10.1111/j.1540-6261.1996.tb04074.x",
        url="https://doi.org/10.1111/j.1540-6261.1996.tb04074.x",
        notes="Canonical probability-of-informed-trading reference.",
    ),
    "lopezdeprado2016hrp": paper_ref(
        "lopezdeprado2016hrp",
        title="Building Diversified Portfolios that Outperform Out of Sample",
        authors=["Marcos López de Prado"],
        year=2016,
        venue="The Journal of Portfolio Management",
        doi="10.3905/jpm.2016.42.4.059",
        url="https://doi.org/10.3905/jpm.2016.42.4.059",
        notes="Canonical hierarchical risk parity reference.",
    ),
    "lopezdeprado2018afml": paper_ref(
        "lopezdeprado2018afml",
        title="Advances in Financial Machine Learning",
        authors=["Marcos López de Prado"],
        year=2018,
        venue="Wiley",
        url="https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086",
        ref_type="book",
        notes="Used as the broad reference for fractional differentiation and related execution utilities.",
    ),
    "glasserman2003": paper_ref(
        "glasserman2003",
        title="Monte Carlo Methods in Financial Engineering",
        authors=["Paul Glasserman"],
        year=2003,
        venue="Springer",
        doi="10.1007/978-0-387-21617-1",
        url="https://doi.org/10.1007/978-0-387-21617-1",
        ref_type="book",
        notes="Standard quantitative finance Monte Carlo reference.",
    ),
    "gatheral2006": paper_ref(
        "gatheral2006",
        title="The Volatility Surface: A Practitioner's Guide",
        authors=["Jim Gatheral"],
        year=2006,
        venue="Wiley",
        url="https://onlinelibrary.wiley.com/doi/book/10.1002/9781119203643",
        ref_type="book",
        notes="Standard local-volatility and implied-volatility surface reference.",
    ),
    "neal2011hmc": paper_ref(
        "neal2011hmc",
        title="MCMC Using Hamiltonian Dynamics",
        authors=["Radford M. Neal"],
        year=2011,
        venue="Handbook of Markov Chain Monte Carlo",
        url="https://arxiv.org/abs/1206.1901",
        ref_type="book",
        notes="Canonical HMC reference cited by the upstream AdvancedHMC.jl repository.",
    ),
    "hoffman2014nuts": paper_ref(
        "hoffman2014nuts",
        title="The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo",
        authors=["Matthew D. Hoffman", "Andrew Gelman"],
        year=2014,
        venue="Journal of Machine Learning Research",
        url="https://jmlr.org/papers/v15/hoffman14a.html",
        notes="Canonical NUTS reference.",
    ),
    "girolami2011rmhmc": paper_ref(
        "girolami2011rmhmc",
        title="Riemann manifold Langevin and Hamiltonian Monte Carlo methods",
        authors=["Mark Girolami", "Ben Calderhead"],
        year=2011,
        venue="Journal of the Royal Statistical Society: Series B",
        doi="10.1111/j.1467-9868.2010.00765.x",
        url="https://doi.org/10.1111/j.1467-9868.2010.00765.x",
        notes="Canonical RMHMC reference.",
    ),
    "metropolis1953": paper_ref(
        "metropolis1953",
        title="Equation of State Calculations by Fast Computing Machines",
        authors=[
            "Nicholas Metropolis",
            "Arianna W. Rosenbluth",
            "Marshall N. Rosenbluth",
            "Augusta H. Teller",
            "Edward Teller",
        ],
        year=1953,
        venue="The Journal of Chemical Physics",
        doi="10.1063/1.1699114",
        url="https://doi.org/10.1063/1.1699114",
        notes="Foundational Metropolis sampling paper.",
    ),
    "hastings1970": paper_ref(
        "hastings1970",
        title="Monte Carlo sampling methods using Markov chains and their applications",
        authors=["W. K. Hastings"],
        year=1970,
        venue="Biometrika",
        doi="10.1093/biomet/57.1.97",
        url="https://doi.org/10.1093/biomet/57.1.97",
        notes="Foundational Metropolis-Hastings generalization.",
    ),
    "roberts1996mala": paper_ref(
        "roberts1996mala",
        title="Exponential convergence of Langevin distributions and their discrete approximations",
        authors=["Gareth O. Roberts", "Richard L. Tweedie"],
        year=1996,
        venue="Bernoulli",
        url="https://projecteuclid.org/journals/bernoulli/volume-2/issue-4/Exponential-convergence-of-Langevin-distributions-and-their-discrete-approximations/10.2307/3318418.full",
        notes="Standard MALA attribution reference.",
    ),
    "terbraak2006demc": paper_ref(
        "terbraak2006demc",
        title="A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces",
        authors=["Cajo J. F. ter Braak"],
        year=2006,
        venue="Statistics and Computing",
        doi="10.1007/s11222-006-8769-1",
        url="https://doi.org/10.1007/s11222-006-8769-1",
        notes="Canonical differential evolution MCMC reference.",
    ),
    "kou2006equi": paper_ref(
        "kou2006equi",
        title="Equi-energy sampler with applications in statistical inference and statistical mechanics",
        authors=["S. C. Kou", "Q. Zhou", "W. H. Wong"],
        year=2006,
        venue="The Annals of Statistics",
        doi="10.1214/009053606000000614",
        url="https://doi.org/10.1214/009053606000000614",
        notes="Canonical equi-energy sampling reference.",
    ),
    "gordon1993particle": paper_ref(
        "gordon1993particle",
        title="Novel approach to nonlinear/non-Gaussian Bayesian state estimation",
        authors=["Neil J. Gordon", "David J. Salmond", "Adrian F. M. Smith"],
        year=1993,
        venue="IEE Proceedings F - Radar and Signal Processing",
        doi="10.1049/ip-f-2.1993.0015",
        url="https://doi.org/10.1049/ip-f-2.1993.0015",
        notes="Canonical bootstrap particle filter reference.",
    ),
    "kucukelbir2017advi": paper_ref(
        "kucukelbir2017advi",
        title="Automatic Differentiation Variational Inference",
        authors=["Alp Kucukelbir", "Dustin Tran", "Rajesh Ranganath", "Andrew Gelman", "David M. Blei"],
        year=2017,
        venue="Journal of Machine Learning Research",
        url="https://jmlr.org/papers/v18/16-107.html",
        notes="Canonical ADVI reference.",
    ),
    "titsias2014dsvb": paper_ref(
        "titsias2014dsvb",
        title="Doubly Stochastic Variational Bayes for Non-Conjugate Inference",
        authors=["Michalis K. Titsias", "Miguel Lázaro-Gredilla"],
        year=2014,
        venue="International Conference on Machine Learning",
        url="https://proceedings.mlr.press/v32/titsias14.html",
        notes="Canonical reparameterization-gradient VI reference cited by AdvancedVI docs.",
    ),
    "murphy1999loopy": paper_ref(
        "murphy1999loopy",
        title="Loopy Belief Propagation for Approximate Inference: An Empirical Study",
        authors=["Kevin P. Murphy", "Yair Weiss", "Michael I. Jordan"],
        year=1999,
        venue="Proceedings of UAI",
        url="https://www.cs.cmu.edu/~epxing/Class/10708-16/lecture_notes/Murphy99.pdf",
        notes="Canonical loopy belief propagation empirical study.",
    ),
    "clrs2009": paper_ref(
        "clrs2009",
        title="Introduction to Algorithms",
        authors=[
            "Thomas H. Cormen",
            "Charles E. Leiserson",
            "Ronald L. Rivest",
            "Clifford Stein",
        ],
        year=2009,
        venue="MIT Press",
        url="https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/",
        ref_type="book",
        notes="Broad textbook reference for classical algorithm implementations.",
    ),
    "brunton2016sindy": paper_ref(
        "brunton2016sindy",
        title="Discovering governing equations from data by sparse identification of nonlinear dynamical systems",
        authors=["Steven L. Brunton", "Joshua L. Proctor", "J. Nathan Kutz"],
        year=2016,
        venue="Proceedings of the National Academy of Sciences",
        doi="10.1073/pnas.1517384113",
        url="https://doi.org/10.1073/pnas.1517384113",
        notes="Canonical SINDy paper.",
    ),
    "lorimer2005pulsar": paper_ref(
        "lorimer2005pulsar",
        title="Handbook of Pulsar Astronomy",
        authors=["Duncan R. Lorimer", "Michael Kramer"],
        year=2005,
        venue="Cambridge University Press",
        url="https://www.cambridge.org/core/books/handbook-of-pulsar-astronomy/2226D0FA66B5D5A6DD7E830A164AD10D",
        ref_type="book",
        notes="Broad reference for pulsar search, dedispersion, folding, and SNR calculations.",
    ),
    "urban2013almanac": paper_ref(
        "urban2013almanac",
        title="Explanatory Supplement to the Astronomical Almanac",
        authors=["Sean E. Urban", "P. Kenneth Seidelmann"],
        year=2013,
        venue="University Science Books",
        url="https://books.google.com/books?id=XCjjuwm3SXcC",
        ref_type="book",
        notes="Authoritative broad reference for astronomical time scales, Julian dates, and time conversions.",
    ),
    "gueant2013inventory": paper_ref(
        "gueant2013inventory",
        title="Dealing with the Inventory Risk: a solution to the market making problem",
        authors=["Olivier Guéant", "Charles-Albert Lehalle", "Joaquin Fernandez-Tapia"],
        year=2013,
        venue="Mathematics and Financial Economics",
        doi="10.1007/s11579-012-0086-0",
        url="https://doi.org/10.1007/s11579-012-0086-0",
        notes="Canonical GLFT inventory-risk market-making paper.",
    ),
    "huchra1982fof": paper_ref(
        "huchra1982fof",
        title="Groups of galaxies. I. Nearby groups",
        authors=["John P. Huchra", "Margaret J. Geller"],
        year=1982,
        venue="The Astrophysical Journal",
        doi="10.1086/159999",
        url="https://doi.org/10.1086/159999",
        notes="Canonical friends-of-friends clustering reference.",
    ),
    "berman2000pdb": paper_ref(
        "berman2000pdb",
        title="The Protein Data Bank",
        authors=[
            "Helen M. Berman",
            "John Westbrook",
            "Zukang Feng",
            "Gary Gilliland",
            "T. N. Bhat",
            "Helge Weissig",
            "I. N. Shindyalov",
            "Philip E. Bourne",
        ],
        year=2000,
        venue="Nucleic Acids Research",
        doi="10.1093/nar/28.1.235",
        url="https://doi.org/10.1093/nar/28.1.235",
        notes="Canonical Protein Data Bank reference for hPDB parser atoms.",
    ),
    "kissell2013algo": paper_ref(
        "kissell2013algo",
        title="The Science of Algorithmic Trading and Portfolio Management",
        authors=["Robert Kissell"],
        year=2013,
        venue="Academic Press",
        url="https://www.sciencedirect.com/book/9780124016897/the-science-of-algorithmic-trading-and-portfolio-management",
        ref_type="book",
        notes="Broad execution-algorithm reference for local quant_engine atoms without a unique upstream paper.",
    ),
}


def load_reviewed_atoms() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return manifest["reviewed_atoms"]


def normalize_atom_manifest_atom(atom: str) -> str:
    return "ageoa." + atom.replace("/", ".").replace(":", ".")


def load_upstream_repo_map() -> dict[str, str]:
    items = yaml.safe_load(ATOM_MANIFEST_PATH.read_text())
    repo_map: dict[str, str] = {}
    for item in items:
        fqdn = normalize_atom_manifest_atom(item["atom"])
        upstream = item.get("upstream") or {}
        repo = upstream.get("repo")
        if repo:
            repo_map[fqdn] = repo
    return repo_map


def load_repo_urls(repo_names: set[str]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for repo_name in sorted(repo_names):
        repo_dir = ROOT / "third_party" / repo_name
        if not repo_dir.exists():
            continue
        try:
            url = subprocess.check_output(
                ["git", "-C", str(repo_dir), "remote", "get-url", "origin"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        urls[repo_name] = url
    return urls


def repo_ref_id(repo_name: str) -> str:
    slug = repo_name.lower().replace(".", "_").replace("-", "_")
    return f"repo_{slug}"


def add_assignment(assignments: list[dict], ref_id: str, priority: int, confidence: str, notes: str) -> None:
    assignments.append(
        {
            "ref_id": ref_id,
            "priority": priority,
            "match_metadata": {
                "similarity_score": None,
                "match_type": "manual",
                "matched_nodes": [],
                "confidence": confidence,
                "notes": notes,
            },
        }
    )


def assign_refs(entry: dict, upstream_repo: str | None) -> list[dict]:
    fqdn = entry["atom"]
    module_family = entry.get("module_family", "")
    atom_id = entry["atom_id"]
    lower = fqdn.lower()
    assignments: list[dict] = []

    if upstream_repo:
        add_assignment(
            assignments,
            repo_ref_id(upstream_repo),
            priority=90,
            confidence="high",
            notes=f"Upstream implementation is mapped to the vendored {upstream_repo} repository.",
        )

    if module_family == "numpy":
        add_assignment(assignments, "numpy2020", 20, "high", "Canonical NumPy software citation.")
    elif module_family == "scipy":
        add_assignment(assignments, "scipy2020", 20, "high", "Canonical SciPy software citation.")
    elif module_family == "alphafold":
        add_assignment(assignments, "alphafold2021", 10, "high", "Exact AlphaFold attribution.")
    elif module_family == "pronto":
        add_assignment(assignments, "camurri2020pronto", 20, "high", "Primary Pronto software paper.")
    elif module_family == "institutional_quant_engine":
        add_assignment(
            assignments,
            repo_ref_id("Institutional-Quant-Engine"),
            30,
            "high",
            "Primary repository attribution for Institutional-Quant-Engine atoms.",
        )
    elif module_family == "neurokit2":
        add_assignment(assignments, "neurokit2021", 20, "high", "Preferred NeuroKit2 citation.")
    elif module_family == "skyfield":
        add_assignment(assignments, "skyfield2019ascl", 20, "high", "Preferred Skyfield citation.")
    elif module_family == "astroflow":
        add_assignment(assignments, "astroflow2025", 20, "high", "Primary AstroFlow paper.")
    elif module_family == "mint":
        add_assignment(assignments, "mint2026", 20, "high", "Primary mint paper cited by the upstream repository.")
    elif module_family == "particle_filters":
        add_assignment(assignments, "gordon1993particle", 20, "high", "Canonical particle filter paper.")
    elif module_family == "algorithms":
        add_assignment(assignments, "clrs2009", 30, "medium", "Broad textbook attribution for classical algorithms.")
    elif module_family == "datadriven":
        add_assignment(assignments, "brunton2016sindy", 10, "high", "Exact SINDy attribution.")
    elif module_family in {"pulsar", "pulsar_folding"}:
        add_assignment(assignments, "lorimer2005pulsar", 20, "medium", "Broad pulsar-search attribution.")
    elif module_family == "tempo":
        add_assignment(assignments, repo_ref_id("Tempo.jl"), 30, "high", "Local tempo wrappers mirror Tempo.jl time-scale logic.")
        add_assignment(assignments, "urban2013almanac", 20, "medium", "Broad astronomical time-scale attribution.")
    elif module_family == "tempo_jl":
        add_assignment(assignments, repo_ref_id("Tempo.jl"), 20, "high", "Primary Tempo.jl software attribution.")
        add_assignment(assignments, "urban2013almanac", 10, "medium", "Broad astronomical time-scale attribution.")
    elif module_family == "pasqal":
        add_assignment(assignments, "moleculardocking2025", 20, "high", "Pasqal docking atoms derive from the Molecular-Docking workflow.")
    elif module_family == "molecular_docking":
        add_assignment(assignments, "moleculardocking2025", 20, "high", "Primary Molecular-Docking paper.")
    elif module_family == "hftbacktest":
        add_assignment(assignments, "gueant2013inventory", 20, "high", "GLFT-based market-making attribution.")
    elif module_family == "rust_robotics":
        add_assignment(
            assignments,
            repo_ref_id("rust_robotics"),
            30,
            "high",
            "Primary repository attribution for rust_robotics atoms.",
        )
    elif module_family == "belief_propagation":
        add_assignment(assignments, "murphy1999loopy", 10, "high", "Exact loopy belief propagation attribution.")
    elif module_family == "quant_engine":
        add_assignment(assignments, "kissell2013algo", 30, "medium", "Broad execution-algorithm attribution for local quant_engine atoms.")
    elif module_family == "hPDB":
        add_assignment(assignments, "berman2000pdb", 20, "high", "Canonical Protein Data Bank attribution.")

    if module_family == "biosppy":
        add_assignment(assignments, repo_ref_id("BioSPPy"), 30, "high", "Primary BioSPPy software attribution.")
    if module_family == "kalman_filters":
        add_assignment(assignments, "kalman1960", 20, "high", "Canonical Kalman filter attribution.")
    if module_family == "e2e_ppg":
        add_assignment(assignments, "feli2023pipeline", 20, "high", "Primary E2E-PPG pipeline paper.")
    if module_family == "advancedvi":
        add_assignment(assignments, "kucukelbir2017advi", 30, "medium", "Broad ADVI/VI attribution for AdvancedVI atoms.")
    if module_family == "jax_advi":
        add_assignment(assignments, "kucukelbir2017advi", 20, "high", "Exact ADVI attribution for jax_advi atoms.")
    if module_family == "conjugate_priors":
        add_assignment(assignments, "clrs2009", 95, "low", "Fallback broad reference; implementation attribution comes from the upstream repository.")
    if module_family == "bayes_rs":
        add_assignment(assignments, "clrs2009", 95, "low", "Fallback broad reference; implementation attribution comes from the upstream repository.")

    if "ecg_hamilton" in lower:
        add_assignment(assignments, "hamilton1986", 5, "high", "Exact Hamilton detector attribution.")
    if "ecg_engzee" in lower:
        add_assignment(assignments, "engzee1979", 5, "high", "Exact Engelse-Zeelenberg detector attribution.")
    if "ecg_christov" in lower:
        add_assignment(assignments, "christov2004", 5, "high", "Exact Christov detector attribution.")
    if "ecg_zz2018" in lower or lower.endswith("zhao2018hrvanalysis"):
        add_assignment(assignments, "zhao2018ecgsqi", 5, "high", "Exact ECG SQI attribution.")
    if "abp_zong" in lower:
        add_assignment(assignments, "zong2003abp", 5, "high", "Exact ABP onset detector attribution.")
    if "ppg_elgendi" in lower:
        add_assignment(assignments, "elgendi2013ppg", 5, "high", "Exact Elgendi PPG onset attribution.")
    if "ppg_kavsaoglu" in lower:
        add_assignment(assignments, "kavsaoglu2016ppg", 5, "high", "Exact Kavsaoglu PPG onset attribution.")
    if "thresholdbasedsignalsegmentation" in lower or "asi_signal_segmenter" in lower:
        add_assignment(assignments, "gutierrezrivas2015", 5, "high", "Primary adaptive-threshold ECG detector attribution.")
        add_assignment(assignments, "sadhukhan2012", 6, "high", "Secondary ASI detector attribution cited in upstream source.")
    if "emg_bonato" in lower:
        add_assignment(assignments, "bonato1998", 5, "high", "Exact Bonato onset detector attribution.")
    if "emg_abbink" in lower:
        add_assignment(assignments, "abbink1998", 5, "high", "Exact Abbink onset detector attribution.")
    if "emg_solnik" in lower:
        add_assignment(assignments, "solnik2010", 5, "high", "Exact Solnik onset detector attribution.")

    if module_family == "e2e_ppg":
        if "ppg_sqa" in lower:
            add_assignment(assignments, "feli2023sqa", 5, "high", "Exact E2E-PPG signal quality attribution.")
        if any(token in lower for token in ["ppg_reconstruction", "gan_rec", "reconstruction."]):
            add_assignment(assignments, "wang2022ppg", 5, "high", "Exact E2E-PPG reconstruction attribution.")
        if "kazemi" in lower:
            add_assignment(assignments, "kazemi2022ppg", 5, "high", "Exact E2E-PPG peak detection attribution.")

    if module_family in {"institutional_quant_engine", "quant_engine"}:
        if "almgren_chriss" in lower or lower.endswith("execute_vwap") or lower.endswith("execute_pov") or lower.endswith("execute_passive"):
            add_assignment(assignments, "almgren2000", 10, "high", "Execution scheduling attribution.")
        if "avellaneda_stoikov" in lower or "market_making_avellaneda" in lower:
            add_assignment(assignments, "avellaneda2008", 10, "high", "Market-making attribution.")
        if "heston" in lower:
            add_assignment(assignments, "heston1993", 10, "high", "Exact Heston attribution.")
        if "fractional_diff" in lower:
            add_assignment(assignments, "lopezdeprado2018afml", 10, "medium", "Fractional differentiation attribution.")
        if "hierarchical_risk_parity" in lower or ".hrp_" in lower:
            add_assignment(assignments, "lopezdeprado2016hrp", 10, "high", "Exact HRP attribution.")
        if "hawkes" in lower:
            add_assignment(assignments, "hawkes1971", 10, "high", "Exact Hawkes process attribution.")
        if "order_flow_imbalance" in lower or lower.endswith("calculate_ofi"):
            add_assignment(assignments, "cont2014ofi", 10, "high", "Order-flow imbalance attribution.")
        if "pin_" in lower or ".pin_" in lower or lower.endswith("pin_informed_trading"):
            add_assignment(assignments, "easley1996pin", 10, "high", "PIN attribution.")
        if "copula" in lower:
            add_assignment(assignments, "nelsen2006copulas", 10, "medium", "Copula modeling attribution.")
        if "evt_model" in lower:
            add_assignment(assignments, "coles2001evt", 10, "medium", "EVT/GPD tail-fitting attribution.")
        if "dynamic_hedge" in lower or ".kalman_filter." in lower:
            add_assignment(assignments, "kalman1960", 10, "high", "Kalman-based state estimation attribution.")

    if module_family == "quantfin":
        if "montecarlo" in lower or "monte_carlo" in lower or "functional_monte_carlo" in lower:
            add_assignment(assignments, "glasserman2003", 10, "high", "Monte Carlo pricing attribution.")
        if "char_func_option" in lower:
            add_assignment(assignments, "heston1993", 10, "high", "Characteristic-function option pricing attribution.")
        if "local_vol" in lower or "volatility_surface_modeling" in lower:
            add_assignment(assignments, "gatheral2006", 10, "medium", "Local-volatility / vol-surface attribution.")

    if module_family == "mcmc_foundational":
        if "advancedhmc" in lower:
            add_assignment(assignments, "advancedhmc2020", 20, "high", "Upstream AdvancedHMC.jl package paper.")
        if ".hmc." in lower or "hamiltonian" in lower or "phasepoint" in lower:
            add_assignment(assignments, "neal2011hmc", 10, "high", "Exact HMC attribution.")
        if ".nuts." in lower or "nuts_" in lower:
            add_assignment(assignments, "hoffman2014nuts", 10, "high", "Exact NUTS attribution.")
        if "rmhmc" in lower:
            add_assignment(assignments, "girolami2011rmhmc", 10, "high", "Exact RMHMC attribution.")
        if ".rwmh." in lower:
            add_assignment(assignments, "metropolis1953", 10, "high", "Original Metropolis attribution.")
            add_assignment(assignments, "hastings1970", 11, "high", "Metropolis-Hastings generalization attribution.")
        if ".mala." in lower:
            add_assignment(assignments, "roberts1996mala", 10, "medium", "Standard MALA attribution.")
        if ".de." in lower:
            add_assignment(assignments, "terbraak2006demc", 10, "high", "Differential evolution MCMC attribution.")
        if ".aees." in lower:
            add_assignment(assignments, "kou2006equi", 10, "high", "Equi-energy sampler attribution.")

    if module_family == "advancedvi":
        if "repgradelbo" in lower:
            add_assignment(assignments, "titsias2014dsvb", 15, "high", "Reparameterization-gradient VI attribution.")
    if module_family == "jax_advi":
        add_assignment(assignments, "titsias2014dsvb", 25, "medium", "Complementary reparameterization-gradient VI attribution.")

    if module_family == "particle_filters":
        add_assignment(assignments, "gordon1993particle", 10, "high", "Exact particle filtering attribution.")
    if module_family == "belief_propagation":
        add_assignment(assignments, "murphy1999loopy", 5, "high", "Exact loopy belief propagation attribution.")
    if module_family == "hftbacktest":
        add_assignment(assignments, "gueant2013inventory", 10, "high", "Exact GLFT attribution.")
    if module_family == "jFOF" and lower.endswith("find_fof_clusters"):
        add_assignment(assignments, "huchra1982fof", 10, "medium", "Friends-of-friends clustering attribution.")

    deduped: dict[str, dict] = {}
    for item in assignments:
        ref_id = item["ref_id"]
        current = deduped.get(ref_id)
        if current is None or (item["priority"], ref_id) < (current["priority"], ref_id):
            deduped[ref_id] = item

    ordered = sorted(deduped.values(), key=lambda item: (item["priority"], item["ref_id"]))
    return [{"ref_id": item["ref_id"], "match_metadata": item["match_metadata"]} for item in ordered]


def main() -> None:
    reviewed_atoms = load_reviewed_atoms()
    upstream_repo_map = load_upstream_repo_map()

    repo_names = set(upstream_repo_map.values()) | {"Tempo.jl"}
    repo_urls = load_repo_urls(repo_names)

    registry_refs = dict(MANUAL_REFS)
    for repo_name, repo_url in repo_urls.items():
        registry_refs[repo_ref_id(repo_name)] = repo_ref(repo_ref_id(repo_name), repo_name, repo_url)

    grouped: dict[Path, dict[str, dict]] = defaultdict(dict)
    covered = 0
    uncovered: list[str] = []

    for entry in reviewed_atoms:
        atom_id = entry["atom_id"]
        atom_path = Path(entry["path"])
        atom_dir = atom_path.parent
        upstream_repo = upstream_repo_map.get(entry["atom"])
        refs = assign_refs(entry, upstream_repo)
        if refs:
            covered += 1
        else:
            uncovered.append(atom_id)
        grouped[atom_dir][atom_id] = {
            "references": refs,
            "auto_attribution_runs": [],
        }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps({"schema_version": "1.0", "references": dict(sorted(registry_refs.items()))}, indent=2) + "\n"
    )

    for rel_dir, atoms in sorted(grouped.items(), key=lambda item: str(item[0])):
        out_path = ROOT / rel_dir / "references.json"
        payload = {
            "schema_version": "1.1",
            "atoms": {atom_id: atoms[atom_id] for atom_id in sorted(atoms)},
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Reviewed atoms: {len(reviewed_atoms)}")
    print(f"Covered atoms: {covered}")
    print(f"Registry references: {len(registry_refs)}")
    print(f"Reference files written: {len(grouped)}")
    if uncovered:
        print("Uncovered atoms:")
        for atom_id in uncovered:
            print(atom_id)


if __name__ == "__main__":
    main()
