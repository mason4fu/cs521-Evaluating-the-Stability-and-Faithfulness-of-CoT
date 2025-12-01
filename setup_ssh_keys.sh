#!/usr/bin/env bash
# Setup SSH keys for passwordless login to VM
set -euo pipefail

VM_USER="exouser"
VM_HOST="149.165.151.46"
SSH_KEY="${HOME}/.ssh/id_ed25519.pub"

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH public key not found at $SSH_KEY"
    exit 1
fi

echo "🔑 Setting up SSH key authentication..."
echo "📋 Your public key:"
cat "$SSH_KEY"
echo ""
echo "🔐 Please enter your VM password when prompted..."

# Copy SSH key to VM
ssh-copy-id -i "$SSH_KEY" "${VM_USER}@${VM_HOST}" 2>/dev/null || {
    echo "⚠️  ssh-copy-id failed. Manual setup:"
    echo ""
    echo "1. SSH to VM: ssh ${VM_USER}@${VM_HOST}"
    echo "2. Run these commands on the VM:"
    echo "   mkdir -p ~/.ssh"
    echo "   chmod 700 ~/.ssh"
    echo "   echo '$(cat $SSH_KEY)' >> ~/.ssh/authorized_keys"
    echo "   chmod 600 ~/.ssh/authorized_keys"
}

echo "✅ SSH key setup complete!"
echo "🧪 Testing passwordless login..."
ssh -o BatchMode=yes -o ConnectTimeout=5 "${VM_USER}@${VM_HOST}" "echo '✅ Passwordless login works!'" && {
    echo "✅ Passwordless login confirmed!"
} || {
    echo "⚠️  Passwordless login not working yet. You may need to set up the key manually."
}

