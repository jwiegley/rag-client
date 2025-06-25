{
   description = "llama-index RAG tool";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      version = builtins.substring 0 8 self.lastModifiedDate;
      pkgs = import nixpkgs { inherit system; };

      pythonPackage = pkgs.python3.withPackages (
        python-pkgs: with python-pkgs; [
          pip
          setuptools
          wheel
        ]
      );

      # Required system libraries that pip-installed packages might need
      sysLibs = with pkgs; [
        stdenv.cc.cc.lib  # libstdc++
        zlib
        glib
        libpq
        libpq.pg_config
        openssl
      ];

      libPath = pkgs.lib.makeLibraryPath sysLibs;
    in {
      inherit pkgs;

      packages.default = with pkgs; python3Packages.buildPythonApplication {
        pname = "rag-client";
        version = "1.0.0";
        format = "setuptools";

        src = ./.;

        entryPoints = "main:main";

        # Skip Nix dependency checks
        doCheck = false;
        # pythonImportsCheck = [ "rag" ];
        pythonImportsCheck = [ ];

        nativeBuildInputs = [
          makeWrapper
        ];

        buildInputs = [
          pythonPackage
          zlib
          libpq
          libpq.pg_config
          openssl.dev
        ];

        preBuild = ''
          # Create temporary build environment
          export BUILD_VENV=$TMPDIR/build_venv
          ${pythonPackage}/bin/python -m venv $BUILD_VENV
          source $BUILD_VENV/bin/activate

          export PATH=${libpq.pg_config}/bin:$PATH

          # Install dependencies using pip
          pip install --upgrade pip
          pip install -r requirements.txt
        '';

        installPhase = ''
          # Create output directories
          mkdir -p $out/lib $out/bin $out/venv

          # Install application without dependencies
          ${pythonPackage}/bin/python setup.py install --prefix=$out

          # Clone build venv to output directory
          cp -r $BUILD_VENV $out/venv

          cat > compileall <<EOF
          import compileall
          import sys
          import pathlib
          # Path to your venv's site-packages
          site_packages = pathlib.Path("$out") \
              / "venv" \
              / "build_venv" \
              / "lib" \
              / "python${pythonPackage.pythonVersion}" \
              / "site-packages"
          compileall.compile_dir(site_packages, force=True)
          EOF

          source $out/venv/build_venv/bin/activate
          ${pythonPackage}/bin/python compileall

          # Wrap executables to use local virtual environment
          cat > $out/bin/rag-client <<EOF
          #!${pkgs.bash}/bin/bash
          source $out/venv/build_venv/bin/activate
          export LD_LIBRARY_PATH=${pythonPackage}/lib:${libPath}:$LD_LIBRARY_PATH
          export DYLD_LIBRARY_PATH=${pythonPackage}/lib:${libPath}:$DYLD_LIBRARY_PATH
          export PYTHONPATH="$out/venv/build_venv/lib/python${pythonPackage.pythonVersion}/site-packages:$out/lib/python${pythonPackage.pythonVersion}/site-packages"
          export LD_LIBRARY_PATH=$out/venv/build_venv/lib/python${pythonPackage.pythonVersion}/site-packages/lib:$LD_LIBRARY_PATH
          export DYLD_LIBRARY_PATH=$out/venv/build_venv/lib/python${pythonPackage.pythonVersion}/site-packages/lib:$DYLD_LIBRARY_PATH
          egg=$out/lib/python${pythonPackage.pythonVersion}/site-packages/*.egg
          $out/venv/build_venv/bin/python -c "import sys; \
            sys.path.insert(0, '$egg'); \
            from main import main, parse_args; \
            main(parse_args())" "\$@"
          EOF
        '';
      };

      apps.default = {
        type = "app";
        program = "${self.packages.${system}.default}/bin/rag-client";
      };

      devShells.default = with pkgs; mkShell {
        buildInputs = [ python3 ] ++ sysLibs;

        shellHook = ''
          # Create and activate virtual environment
          [ ! -d .venv ] && python -m venv .venv
          source .venv/bin/activate
          # Update pip and install dependencies
          pip install -U pip
          [ -f requirements.txt ] && pip install -r requirements.txt
          # Make sure we find the libraries
          export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH
          export DYLD_LIBRARY_PATH=${libPath}:$DYLD_LIBRARY_PATH
          export PYTHONPATH="$PWD:$PYTHONPATH"
        '';

        nativeBuildInputs = [
          black                 # Python code formatter
          basedpyright          # LSP server for Python
          isort                 # Sorts imports
          autoflake             # Removes unused imports
          pylint
          cmake
          pkg-config
        ];
      };
    });
}
