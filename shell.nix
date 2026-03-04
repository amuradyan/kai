{ pkgs ? import <nixpkgs> {} }:

let
  # Use a pinned nixpkgs version with compatible Node.js
  nixpkgs-stable = fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.05.tar.gz";
  pkgs-stable = import nixpkgs-stable {};

  # CUDA-enabled Python packages
  cudaPackages = pkgs.cudaPackages_12;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python
    python312
    python312Packages.pip
    python312Packages.virtualenv

    # Build tools
    gcc14
    git
    cmake
    pkg-config

    # Node.js (using LTS version with compatible dependencies)
    pkgs-stable.nodejs_18

    # CUDA support
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    linuxPackages.nvidia_x11

    # RISC-V cross-compilation
    pkgsCross.riscv64.stdenv.cc

    # Binary analysis tools (system level)
    binutils
    gdb

    # RISC-V emulation
    qemu
  ];

  shellHook = ''
    # Set CUDA paths for PyTorch
    export CUDA_HOME=${cudaPackages.cudatoolkit}
    export CUDA_PATH=${cudaPackages.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.gcc14.cc.lib}/lib:${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export EXTRA_LDFLAGS="-L${pkgs.gcc14.cc.lib}/lib"

    # Make RISC-V toolchain accessible
    export RISCV_GCC=$(which riscv64-unknown-linux-gnu-gcc)

    echo "Nix environment loaded"
    echo "CUDA: $CUDA_HOME"
    echo "RISC-V GCC: $RISCV_GCC"
  '';
}
