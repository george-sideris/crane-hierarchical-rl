#!/bin/bash
# Control-channel entrypoint for RunPod pods.
#
# The isaac-sim base image sets ENTRYPOINT ["/bin/sh","-c","/isaac-sim/runheadless.sh"]. Because
# that is the sh -c form, anything RunPod passes as a start command arrives as ignored positional
# parameters ($0, $1, ...) and the pod boots straight into the Isaac Sim app with no shell and no
# sshd. Verified on pod winvxsndlrvrj3, 2026-08-09. RunPod's v2 API has no entrypoint field, so
# the fix has to live in the image: this script replaces that entrypoint.
#
# It authorizes the pod's injected key and then runs sshd in the FOREGROUND, which both gives the
# control channel and keeps the container alive (the container dies when PID 1 exits).
set -u

mkdir -p /root/.ssh /var/run/sshd
chmod 700 /root/.ssh

# RunPod injects the pod's authorized key as PUBLIC_KEY.
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
else
  echo "WARNING: PUBLIC_KEY is empty - no key authorized, ssh will refuse you." >&2
fi

# Pod env vars are injected into PID 1 only, so an ssh session does not see them and Isaac Sim
# would block on the unaccepted EULA. /etc/environment is the one that works for NON-interactive
# ssh ("ssh host cmd"): pam_env applies it at session setup, whereas /etc/profile.d is only
# sourced by login shells and ~/.bashrc returns early for non-interactive shells on Ubuntu.
KEEP='^(ACCEPT_EULA|PRIVACY_CONSENT|ISAACLAB_PATH|ISAACSIM_PATH|OMNI_KIT_ALLOW_ROOT)='
touch /etc/environment
{ grep -vE "$KEEP" /etc/environment 2>/dev/null || true; env | grep -E "$KEEP" || true; } \
  > /tmp/environment.new && mv /tmp/environment.new /etc/environment
chmod 644 /etc/environment
# Interactive shells too, for anyone who lands on the pod by hand.
env | grep -E "$KEEP" | sed 's/^\([^=]*\)=\(.*\)$/export \1="\2"/' \
  > /etc/profile.d/runpod_env.sh 2>/dev/null || true
chmod 644 /etc/profile.d/runpod_env.sh 2>/dev/null || true

# Anything passed as a start command runs in the background, so it cannot take the container down
# with it and cannot block sshd from coming up.
if [ "$#" -gt 0 ]; then
  echo "running start command in background: $*"
  setsid bash -c "$*" > /var/log/runpod_startcmd.log 2>&1 < /dev/null &
fi

exec /usr/sbin/sshd -D -e
