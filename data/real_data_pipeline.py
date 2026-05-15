import os
import json
import time
import logging
import hashlib
import requests
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent / "real"

class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

class MaterialsProjectCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("MP_API_KEY", "")
        self.base_url = "https://api.materialsproject.org"
        self.limiter = RateLimiter(calls_per_second=5)
        self.save_dir = BASE_DIR / "materials_project"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-API-KEY": self.api_key})

    def _get(self, endpoint, params=None):
        self.limiter.wait()
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"MP API error on {endpoint}: {e}")
            return None

    def fetch_battery_cathodes(self):
        log.info("Fetching Na-ion cathode materials from Materials Project...")
        systems = [
            "Na-Mn-O", "Na-Fe-O", "Na-Mn-Fe-O", "Na-Co-O", "Na-Ni-O",
            "Na-Mn-Ni-O", "Na-Fe-Mn-Ni-O", "Na-Ti-O", "Na-V-O",
            "Na-Cr-O", "Na-Cu-O", "Na-Mn-Al-O", "Na-Mn-Ti-O", "Na-Mn-Mg-O",
            "Li-Co-O", "Li-Mn-O", "Li-Fe-P-O", "Li-Ni-Mn-Co-O", "Li-Ni-Co-Al-O",
            "Li-Fe-O", "Li-Ti-O", "Li-V-O", "Li-Mn-Ni-O"
        ]
        all_materials = []
        for system in systems:
            log.info(f"  Querying system: {system}")
            data = self._get("/v2/materials/summary/", params={
                "chemsys": system,
                "fields": "material_id,formula_pretty,formation_energy_per_atom,band_gap,energy_above_hull,density,volume,nsites,symmetry,composition,theoretical",
                "_limit": 500
            })
            if data and "data" in data:
                entries = data["data"]
                log.info(f"    Found {len(entries)} entries for {system}")
                all_materials.extend(entries)
            time.sleep(0.5)

        df = pd.DataFrame(all_materials)
        if not df.empty:
            df.to_csv(self.save_dir / "cathode_materials.csv", index=False)
            df.to_parquet(self.save_dir / "cathode_materials.parquet", index=False)
            log.info(f"Saved {len(df)} cathode materials to disk")
        return df

    def fetch_electrode_properties(self):
        log.info("Fetching electrode insertion data...")
        data = self._get("/v2/insertion_electrodes/", params={
            "working_ion": "Na",
            "fields": "battery_id,formula_charge,formula_discharge,max_voltage,min_voltage,average_voltage,capacity_grav,capacity_vol,energy_grav,energy_vol,fracA_charge,fracA_discharge,stability_charge,stability_discharge,num_steps",
            "_limit": 1000
        })
        if data and "data" in data:
            df = pd.DataFrame(data["data"])
            df.to_csv(self.save_dir / "na_ion_electrodes.csv", index=False)
            log.info(f"Saved {len(df)} Na-ion electrode entries")

        data_li = self._get("/v2/insertion_electrodes/", params={
            "working_ion": "Li",
            "fields": "battery_id,formula_charge,formula_discharge,max_voltage,min_voltage,average_voltage,capacity_grav,capacity_vol,energy_grav,energy_vol,fracA_charge,fracA_discharge,stability_charge,stability_discharge,num_steps",
            "_limit": 2000
        })
        if data_li and "data" in data_li:
            df_li = pd.DataFrame(data_li["data"])
            df_li.to_csv(self.save_dir / "li_ion_electrodes.csv", index=False)
            log.info(f"Saved {len(df_li)} Li-ion electrode entries")

    def fetch_thermodynamic_data(self):
        log.info("Fetching thermodynamic stability data...")
        elements_of_interest = ["Na", "Mn", "Fe", "Ni", "Co", "Li", "O", "P", "V", "Ti", "Al", "Mg"]
        for el in elements_of_interest:
            data = self._get("/v2/materials/summary/", params={
                "elements": el,
                "fields": "material_id,formula_pretty,formation_energy_per_atom,energy_above_hull,band_gap,is_stable",
                "_limit": 500
            })
            if data and "data" in data:
                df = pd.DataFrame(data["data"])
                df.to_csv(self.save_dir / f"thermo_{el}.csv", index=False)
                log.info(f"  Saved {len(df)} entries containing {el}")

    def fetch_all(self):
        self.fetch_battery_cathodes()
        self.fetch_electrode_properties()
        self.fetch_thermodynamic_data()

class BatteryArchiveCollector:
    def __init__(self):
        self.base_url = "https://www.batteryarchive.org/api"
        self.save_dir = BASE_DIR / "battery_archive"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(calls_per_second=2)
        self.session = requests.Session()

    def _get(self, endpoint, params=None):
        self.limiter.wait()
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"BatteryArchive API error: {e}")
            return None

    def fetch_cell_list(self):
        log.info("Fetching cell list from BatteryArchive...")
        data = self._get("cells")
        if data:
            df = pd.DataFrame(data)
            df.to_csv(self.save_dir / "cell_list.csv", index=False)
            log.info(f"Found {len(df)} cells in BatteryArchive")
            return df
        return pd.DataFrame()

    def fetch_cycle_data(self, cell_id):
        data = self._get(f"cells/{cell_id}/cycles")
        if data:
            df = pd.DataFrame(data)
            df.to_csv(self.save_dir / f"cycles_{cell_id}.csv", index=False)
            return df
        return pd.DataFrame()

    def fetch_timeseries(self, cell_id, cycle_num):
        data = self._get(f"cells/{cell_id}/cycles/{cycle_num}/timeseries")
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()

    def bulk_download(self, max_cells=200):
        cells = self.fetch_cell_list()
        if cells.empty:
            log.warning("No cells found in BatteryArchive. Trying alternative endpoints...")
            return
        cell_ids = cells.iloc[:max_cells]["cell_id"].tolist() if "cell_id" in cells.columns else []
        log.info(f"Downloading cycle data for {len(cell_ids)} cells...")
        for cid in cell_ids:
            self.fetch_cycle_data(cid)

class NASABatteryCollector:
    def __init__(self):
        self.save_dir = BASE_DIR / "nasa_ames"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.urls = {
            "randomized": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
            "prognostics": "https://phm-datasets.s3.amazonaws.com/NASA/11.+Li-ion+Battery+Aging+Datasets.zip"
        }

    def download_dataset(self, name, url):
        dest = self.save_dir / f"{name}.zip"
        if dest.exists():
            log.info(f"NASA {name} already downloaded")
            return
        log.info(f"Downloading NASA {name} dataset ({url})...")
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            log.info(f"Downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            import zipfile
            with zipfile.ZipFile(dest, 'r') as zf:
                zf.extractall(self.save_dir / name)
            log.info(f"Extracted {name}")
        except Exception as e:
            log.error(f"Failed to download NASA {name}: {e}")

    def download_all(self):
        for name, url in self.urls.items():
            self.download_dataset(name, url)

class OxfordBatteryCollector:
    def __init__(self):
        self.save_dir = BASE_DIR / "oxford"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.url = "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/r02870v86s"

    def download(self):
        dest = self.save_dir / "oxford_battery.zip"
        if dest.exists():
            log.info("Oxford dataset already downloaded")
            return
        log.info("Downloading Oxford Battery Degradation Dataset...")
        try:
            resp = requests.get(self.url, stream=True, timeout=600)
            resp.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            log.info(f"Downloaded Oxford dataset ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            log.error(f"Oxford download failed: {e}")

class CALCECollector:
    def __init__(self):
        self.save_dir = BASE_DIR / "calce"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self):
        log.info("CALCE data requires manual download from https://web.calce.umd.edu/batteries/data.htm")
        log.info("Place .xlsx or .csv files in: " + str(self.save_dir))

class KaggleBatteryCollector:
    def __init__(self):
        self.save_dir = BASE_DIR / "kaggle"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def download_datasets(self):
        datasets = [
            "patrickfleith/nasa-battery-dataset",
            "ignaciovinuales/battery-remaining-useful-life-rul",
            "manuelalexandreribeiro/battery-discharge",
        ]
        for ds in datasets:
            log.info(f"To download Kaggle dataset: kaggle datasets download -d {ds} -p {self.save_dir}")
        log.info("Ensure `kaggle` CLI is configured with your API token (~/.kaggle/kaggle.json)")

class HuggingFaceBatteryCollector:
    def __init__(self):
        self.save_dir = BASE_DIR / "huggingface"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def download_datasets(self):
        log.info("Searching HuggingFace for battery datasets...")
        try:
            resp = requests.get("https://huggingface.co/api/datasets", params={"search": "battery degradation", "limit": 20}, timeout=30)
            if resp.ok:
                datasets = resp.json()
                for ds in datasets:
                    ds_id = ds.get("id", "unknown")
                    log.info(f"  Found HF dataset: {ds_id}")
                    try:
                        from datasets import load_dataset
                        data = load_dataset(ds_id, split="train")
                        df = data.to_pandas()
                        safe_name = ds_id.replace("/", "_")
                        df.to_parquet(self.save_dir / f"{safe_name}.parquet", index=False)
                        log.info(f"  Saved {len(df)} rows from {ds_id}")
                    except Exception as e:
                        log.warning(f"  Could not load {ds_id}: {e}")
        except Exception as e:
            log.error(f"HuggingFace search failed: {e}")

class AFLOWCollector:
    def __init__(self):
        self.base_url = "http://aflowlib.duke.edu/API/aflux/"
        self.save_dir = BASE_DIR / "aflow"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(calls_per_second=1)

    def query_battery_materials(self):
        log.info("Querying AFLOW for battery-relevant materials...")
        queries = [
            "species(Na,Mn,O),nspecies(3),paging(1,500)",
            "species(Na,Fe,O),nspecies(3),paging(1,500)",
            "species(Li,Co,O),nspecies(3),paging(1,500)",
            "species(Li,Fe,P,O),nspecies(4),paging(1,500)",
            "species(Li,Ni,Mn,Co,O),nspecies(5),paging(1,500)",
        ]
        all_entries = []
        for q in queries:
            self.limiter.wait()
            url = f"{self.base_url}?{q},format(json)"
            try:
                resp = requests.get(url, timeout=60)
                if resp.ok:
                    data = resp.json()
                    if isinstance(data, list):
                        all_entries.extend(data)
                        log.info(f"  AFLOW returned {len(data)} entries for query")
            except Exception as e:
                log.warning(f"  AFLOW query failed: {e}")

        if all_entries:
            df = pd.DataFrame(all_entries)
            df.to_csv(self.save_dir / "aflow_battery_materials.csv", index=False)
            log.info(f"Saved {len(df)} AFLOW entries")

class OQMDCollector:
    def __init__(self):
        self.base_url = "http://oqmd.org/oqmdapi/formationenergy"
        self.save_dir = BASE_DIR / "oqmd"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(calls_per_second=1)

    def query_compositions(self):
        log.info("Querying OQMD for Na/Li transition metal oxides...")
        filters = [
            "Na-Mn-O", "Na-Fe-O", "Li-Co-O", "Li-Mn-O", "Li-Fe-P-O", "Li-Ni-O"
        ]
        all_data = []
        for f in filters:
            self.limiter.wait()
            try:
                resp = requests.get(self.base_url, params={
                    "composition": f, "fields": "name,entry_id,delta_e,stability,band_gap",
                    "limit": 500, "format": "json"
                }, timeout=60)
                if resp.ok:
                    data = resp.json()
                    if "data" in data:
                        all_data.extend(data["data"])
                        log.info(f"  OQMD returned {len(data['data'])} for {f}")
            except Exception as e:
                log.warning(f"  OQMD query failed for {f}: {e}")

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(self.save_dir / "oqmd_battery_materials.csv", index=False)
            log.info(f"Saved {len(df)} OQMD entries")

class MasterRealDataPipeline:
    def __init__(self, mp_api_key=None):
        self.collectors = {
            "materials_project": MaterialsProjectCollector(api_key=mp_api_key),
            "battery_archive": BatteryArchiveCollector(),
            "nasa": NASABatteryCollector(),
            "oxford": OxfordBatteryCollector(),
            "calce": CALCECollector(),
            "kaggle": KaggleBatteryCollector(),
            "huggingface": HuggingFaceBatteryCollector(),
            "aflow": AFLOWCollector(),
            "oqmd": OQMDCollector(),
        }

    def run_all(self):
        log.info("=" * 60)
        log.info("MASTER REAL DATA PIPELINE - Collecting from ALL sources")
        log.info("=" * 60)

        self.collectors["materials_project"].fetch_all()
        self.collectors["battery_archive"].bulk_download(max_cells=200)
        self.collectors["nasa"].download_all()
        self.collectors["oxford"].download()
        self.collectors["calce"].download_all()
        self.collectors["kaggle"].download_datasets()
        self.collectors["huggingface"].download_datasets()
        self.collectors["aflow"].query_battery_materials()
        self.collectors["oqmd"].query_compositions()

        log.info("=" * 60)
        log.info("REAL DATA COLLECTION COMPLETE")
        self._print_summary()
        log.info("=" * 60)

    def _print_summary(self):
        total_files = 0
        total_bytes = 0
        for subdir in BASE_DIR.rglob("*"):
            if subdir.is_file():
                total_files += 1
                total_bytes += subdir.stat().st_size
        log.info(f"Total files collected: {total_files}")
        log.info(f"Total data size: {total_bytes / 1e6:.1f} MB")
        for source_dir in sorted(BASE_DIR.iterdir()):
            if source_dir.is_dir():
                n = sum(1 for f in source_dir.rglob("*") if f.is_file())
                s = sum(f.stat().st_size for f in source_dir.rglob("*") if f.is_file())
                log.info(f"  {source_dir.name}: {n} files, {s/1e6:.1f} MB")

if __name__ == "__main__":
    mp_key = os.environ.get("MP_API_KEY", "")
    if not mp_key:
        log.warning("No MP_API_KEY set. Materials Project queries will fail.")
        log.warning("Get a free key at https://materialsproject.org/api")
    pipeline = MasterRealDataPipeline(mp_api_key=mp_key)
    pipeline.run_all()
