# Alliance Cluster Notes

## Account

- Username: `jiaqi217`
- CCI: `usz-200`
- CCRI / role: `usz-200-03`
- Institution: University of Toronto
- Department: Computer Science
- Role expires: `2027-06-15`
- Sponsor: Dehan Kong
- Sponsor id shown in role: `kbk-464-01`

## Resource Allocation Project

- RAPI: `kbk-464-aa`
- Group name / likely Slurm account: `def-kdhkdh`
- Status: Active
- Membership: member
- Manager: no
- Owner: no

## Cluster

- Cluster in use: `fir`
- Also available / easier layout observed: `narval`

## Narval Layout

Observed on `narval1`:

```text
$HOME/projects/
  def-annielee/
  def-inghaw/
  def-kdhkdh/
  def-rudner/

$HOME/projects/def-kdhkdh/
  ... jiaqi217 ...
```

For Narval, use the existing personal project directory:

```bash
cd ~/projects/def-kdhkdh/jiaqi217
git clone https://github.com/LOFIBOY217/Repr_Research_Project.git
cd Repr_Research_Project
```

Clone completed successfully on Narval at:

```text
/home/jiaqi217/projects/def-kdhkdh/jiaqi217/Repr_Research_Project
```

Recommended Narval variables:

```bash
export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export REPR_SCRATCH=$SCRATCH/Repr_Research_Project
export SLURM_ACCOUNT=def-kdhkdh
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT
```

## ImageNet

Alliance's centrally provided ImageNet copy is documented as being available on
Nibi only, under `/datashare/imagenet/`, and requires opt-in access after
acknowledging the ImageNet license. Do not assume Narval has a system-wide
ImageNet copy.

To request Nibi ImageNet access:

1. Open the Alliance ImageNet documentation page.
2. In the `Request access through the opt-in service` section, click
   `this opt-in page`.
3. Log in with the Alliance/CCDB account.
4. Select/acknowledge the ImageNet license agreement.
5. Wait for access to propagate, then open a new Nibi shell and test:

```bash
ls /datashare/imagenet
ls /datashare/imagenet/ILSVRC2012
```

On Narval, first check whether the group already has a local copy:

```bash
find ~/projects/def-kdhkdh -maxdepth 4 \( -iname '*imagenet*' -o -iname 'ILSVRC*' \) 2>/dev/null
find ~/projects/def-annielee ~/projects/def-inghaw ~/projects/def-rudner -maxdepth 4 \( -iname '*imagenet*' -o -iname 'ILSVRC*' \) 2>/dev/null
```

For this repository, the expected ImageFolder-style ImageNet path should contain
`train/` and `val/` subdirectories and should be exported as:

```bash
export DATA_ROOT=/path/to/imagenet
```

## Repository

- GitHub repo: `https://github.com/LOFIBOY217/Repr_Research_Project`
- Recommended code location on `narval`: `~/projects/def-kdhkdh/jiaqi217/Repr_Research_Project`
- Possible code location on `fir`: `/project/def-kdhkdh/<writable-subdir>/Repr_Research_Project`

Clone on `fir`:

```bash
cd /project/def-kdhkdh
git clone https://github.com/LOFIBOY217/Repr_Research_Project.git
cd Repr_Research_Project
```

If cloning directly into `/project/def-kdhkdh` fails, do not assume the repo
should live at the project root. On this account, `/project/def-kdhkdh` contains
subdirectories such as `kdhkdh`, `cindyut6`, `dcdecker`, and others. Check which
subdirectory is intended for your work before writing there:

```bash
ls -ld /project/def-kdhkdh /project/def-kdhkdh/kdhkdh
groups
```

If the group expects work under the sponsor/project subdirectory `kdhkdh`, clone
there:

```bash
cd /project/def-kdhkdh/kdhkdh
git clone https://github.com/LOFIBOY217/Repr_Research_Project.git
cd Repr_Research_Project
```

Otherwise, prefer keeping the source code in `$HOME` because the repository is
small, and put large assets, checkpoints, caches, and job outputs in `$SCRATCH`:

```bash
cd $HOME
git clone https://github.com/LOFIBOY217/Repr_Research_Project.git
cd Repr_Research_Project
```

Use scratch paths for large files:

```bash
mkdir -p $SCRATCH/Repr_Research_Project/{checkpoints,data,outputs,hf_cache}
```

If `$HOME` quota is also tight, then use scratch as a temporary working location:

```bash
cd $SCRATCH
git clone https://github.com/LOFIBOY217/Repr_Research_Project.git
cd Repr_Research_Project
```

Scratch is suitable for active work, checkpoints, and temporary outputs, but it
is not backed up and may be purged. Push code changes to GitHub regularly.

## Suggested Shell Variables

The older Compute Canada tutorial uses `~/projects/...` with plural `projects`.
On `fir`, the project filesystem is more likely exposed as singular `~/project/...`
or directly under `/project/...`. Confirm the real path before adding variables to
`~/.bashrc`.

First check:

```bash
echo $PROJECT
echo $SCRATCH
ls -ld ~/project ~/projects ~/scratch /project/def-kdhkdh 2>/dev/null
```

If `/project/def-kdhkdh` exists, use:

```bash
export project=/project/def-kdhkdh
export SLURM_ACCOUNT=def-kdhkdh
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT
```

Then reload:

```bash
source ~/.bashrc
```

After cloning the repo, optionally add:

```bash
export REPR_REPO=$project/Repr_Research_Project
```

## Verification Commands

Run these on the `fir` login node:

```bash
echo $PROJECT
echo $SCRATCH
ls -ld ~/project ~/projects ~/scratch /project/def-kdhkdh 2>/dev/null
echo $project
echo $SLURM_ACCOUNT
cd $project
pwd
```

If `/project/def-kdhkdh` does not exist, look for the actual project path:

```bash
ls /project
```

Use `scratch` only for temporary job outputs and large active reads/writes. It is
not backed up and can be purged.
