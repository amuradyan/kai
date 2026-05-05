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
    # Python with development headers
    python312
    python312Packages.pip
    python312Packages.virtualenv

    # Build tools
    gcc14
    git
    cmake
    pkg-config
    zlib

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
    export LD_LIBRARY_PATH=${pkgs.gcc14.cc.lib}/lib:${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export EXTRA_LDFLAGS="-L${pkgs.gcc14.cc.lib}/lib"

    # Python development headers for Triton compilation
    export C_INCLUDE_PATH=${pkgs.python312}/include/python3.12:''${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
    export CPLUS_INCLUDE_PATH=${pkgs.python312}/include/python3.12:''${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}

    # Triton + NixOS compatibility
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
    export LIBRARY_PATH=/run/opengl-driver/lib:''${LIBRARY_PATH:+:$LIBRARY_PATH}

    # Make RISC-V toolchain accessible
    export RISCV_GCC=$(which riscv64-unknown-linux-gnu-gcc)

    echo "Nix environment loaded"
    echo "CUDA: $CUDA_HOME"
    echo "Python headers: ${pkgs.python312}/include/python3.12"
    echo "TRITON_LIBCUDA_PATH: $TRITON_LIBCUDA_PATH"
    echo "RISC-V GCC: $RISCV_GCC"
  '';
}
