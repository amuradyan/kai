{ pkgs ? import <nixpkgs> {} }:

let
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
    gcc
    git
    cmake
    pkg-config

    # CUDA support
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    linuxPackages.nvidia_x11

    # RISC-V cross-compilation
    pkgsCross.riscv64.stdenv.cc

    # Binary analysis tools (system level)
    binutils
    gdb
  ];

  shellHook = ''
    # Set CUDA paths for PyTorch
    export CUDA_HOME=${cudaPackages.cudatoolkit}
    export CUDA_PATH=${cudaPackages.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export EXTRA_LDFLAGS="-L${pkgs.stdenv.cc.cc.lib}/lib"

    # Make RISC-V toolchain accessible
    export RISCV_GCC=$(which riscv64-unknown-linux-gnu-gcc)

    echo "Nix environment loaded"
    echo "CUDA: $CUDA_HOME"
    echo "RISC-V GCC: $RISCV_GCC"
  '';
}
