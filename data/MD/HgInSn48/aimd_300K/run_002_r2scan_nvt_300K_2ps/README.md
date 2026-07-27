# HgInSn 48-Atom r2SCAN NVT, 300 K

Run type: 2 ps reference, 300 K, 2000 steps, 1 fs timestep.

Files:
- `POSCAR`: selected supercell from `../../structures/HgInSn_48_initial.extxyz`
- `INCAR`: r2SCAN NVT, `NSW=2000`, `POTIM=1 fs`, `T=300 K`
- `KPOINTS`: Gamma-only
- `POTCAR`: copied from the completed smoke-test directory, element order `Hg In_d Sn_d`
- `make_potcar.sh`: optional regeneration helper if pseudopotential permissions are available
- `run.sh`: starts VASP with `NP=${NP:-48}` by default for the 55-core server

Recommended launch commands:

Current 20-core environment, use 16 MPI ranks:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_002_r2scan_nvt_300K_2ps
chmod +x run.sh
NP=16 ./run.sh
```

55-core / 300 GB server, use 48 MPI ranks:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/reference_dft_md/run_002_r2scan_nvt_300K_2ps
chmod +x run.sh
./run.sh
```

Monitor:

```bash
tail -f OSZICAR vasp.out
```

The completed 20-step smoke test averaged about 55 s/step on `NP=16`; 2 ps is therefore roughly 31 hours on this environment before accounting for load variation. On 48 ranks it should be faster, but benchmark the first 20-50 steps.
