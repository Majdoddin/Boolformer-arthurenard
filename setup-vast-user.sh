#!/bin/bash

# Vast.ai User Setup Script
# Run this script as root for steps 1-5, then as ruhollah for step 6+
# Creates user ruhollah with SSH access and installs Claude Code

set -e

USER="ruhollah"
START_STEP=${1:-1}  # Default to step 1 if no argument provided

echo "Setting up user: $USER on Vast.ai instance (starting from step $START_STEP)"

# Ensure user exists (needed for later steps even if we skip user creation)
if [ $START_STEP -gt 5 ] && ! id "$USER" &>/dev/null; then
    echo "Error: User $USER does not exist. Run steps 1-5 as root first."
    exit 1
fi

# Step 1: Create user with sudo privileges (as root)
if [ $START_STEP -le 1 ]; then
    echo "Step 1: Creating user $USER..."
    useradd -m -s /bin/bash -G sudo $USER
fi

# Step 2: Set up passwordless sudo (as root)
if [ $START_STEP -le 2 ]; then
    echo "Step 2: Setting up passwordless sudo..."
    echo "$USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USER
fi

# Step 3: Remove password requirement (as root)
if [ $START_STEP -le 3 ]; then
    echo "Step 3: Removing password requirement..."
    passwd -d $USER
fi

# Step 4: Set up SSH access (as root)
if [ $START_STEP -le 4 ]; then
    echo "Step 4: Setting up SSH access..."
    mkdir -p /home/$USER/.ssh
    cp ~/.ssh/authorized_keys /home/$USER/.ssh/
    chown -R $USER:$USER /home/$USER/.ssh
    chmod 700 /home/$USER/.ssh
    chmod 600 /home/$USER/.ssh/authorized_keys
fi

# Step 5: Install Node.js (as root)
if [ $START_STEP -le 5 ]; then
    echo "Step 5: Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
    apt-get install -y nodejs
fi

# Check if we're running steps 6+ as the correct user
if [ "$(whoami)" != "$USER" ]; then
    echo "Error: Steps 6+ should be run as user '$USER' or root."
    echo "Current user: $(whoami)"
    echo "Please run: ssh $USER@hostname && ./setup-vast-user.sh $START_STEP"
    exit 1
fi

# Step 6: Install git and configure (as root, then user)
if [ $START_STEP -le 6 ]; then
    echo "Step 6: Installing and configuring git..."
    # Install git if not present
    which git || (apt-get update && apt-get install -y git)

    # Configure git as user
    su - $USER -c '
        # Configure git with user credentials
        git config --global user.name "Ruhollah Majdoddin"
        git config --global user.email "r.majdodin@gmail.com"

        # Add GitHub SSH host key
        ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null || (mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts)
    '
fi

# Step 7: Install Claude Code (as user ruhollah)
if [ $START_STEP -le 7 ]; then
    echo "Step 7: Installing Claude Code..."
    su - $USER -c '
        # Create local lib directory
        mkdir -p ~/.local/lib
        cd ~/.local/lib

        # Initialize npm and install Claude Code
        npm init -y
        npm install @anthropic-ai/claude-code

        # Add to PATH
        echo "export PATH=~/.local/lib/node_modules/.bin:\$PATH" >> ~/.bashrc
    '
fi

# Step 8: Clone Boolformer repository (as user)
if [ $START_STEP -le 8 ]; then
    echo "Step 8: Cloning Boolformer repository..."
    su - $USER -c '
        # Clone Boolformer repository from majdoddin
        git clone https://github.com/Majdoddin/Boolformer-arthurenard.git ~/ai/Boolformer
        cd ~/ai/Boolformer

        # Fetch all branches
        git fetch --all

        # List available branches
        echo "Available branches:"
        git branch -a
    '
fi

# Step 9: Set up Python environment with UV (as user)
if [ $START_STEP -le 9 ]; then
    echo "Step 9: Setting up Python environment..."
    su - $USER -c '
        cd ~/ai/Boolformer

        # Install UV
        pip install uv

        # Create virtual environment
        uv venv

        # Install dependencies
        uv pip install -r requirements.txt

        echo "Python environment setup complete!"
        echo "To activate: cd ~/ai/Boolformer && source .venv/bin/activate"
    '
fi

echo "✅ Setup completed successfully!"
echo ""
echo "Summary:"
echo "- User '$USER' created with passwordless sudo"
echo "- SSH access configured"
echo "- Node.js installed"
echo "- Git configured with user credentials"
echo "- Claude Code installed and ready to use"
echo "- Boolformer repository cloned to ~/ai/Boolformer with all branches"
echo "- Python virtual environment created with UV and dependencies installed"
echo ""
echo "You can now connect with: ssh $USER@\$(hostname -I | awk '{print \$1}')"
echo "To use Claude Code, run: claude-code"
echo "To work with Boolformer, run: cd ~/ai/Boolformer && source .venv/bin/activate"
