"""Download + extract CoRE MOF 2019 v2.0.0 (Zenodo record 3677685). Usage: python download_coremof.py [dest_dir]"""
import sys
import mof_data as md

if __name__ == "__main__":
    out = md.download_coremof(sys.argv[1] if len(sys.argv) > 1 else "data")
    cifs = md.find_cifs(out)
    print(f"extracted to {out}; {len(cifs)} CIF files (NCR folders excluded)")
