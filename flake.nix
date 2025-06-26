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
          virtualenv
        ]
      );

      # Required system libraries that pip-installed packages might need
      sysLibs = with pkgs; [
        stdenv.cc.cc.lib  # libstdc++
        zlib
        glib
        libpq
        openssl
        pkg-config
      ];

      libPath = pkgs.lib.makeLibraryPath sysLibs;
    in {
      inherit pkgs;

      packages.default = pkgs.stdenv.mkDerivation {
        pname = "rag-client";
        version = "1.0.0";

        src = ./.;

        nativeBuildInputs = with pkgs; [
          makeWrapper
          pythonPackage
        ];

        buildInputs = sysLibs ++ [
          pkgs.libpq.pg_config
        ];

        buildPhase = ''
          # Create virtual environment
          ${pythonPackage}/bin/python -m venv $out/venv

          # Activate it
          source $out/venv/bin/activate

          # Set environment for building
          export PATH=${pkgs.libpq.pg_config}/bin:$PATH
          export PATH=${pkgs.cmake}/bin:$PATH
          export PATH=${pkgs.pkg-config}/bin:$PATH
          export PATH=${pkgs.git}/bin:$PATH
          export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH

          # Install dependencies
          pip --cache-dir=.pip install --upgrade pip
          pip --cache-dir=.pip install -r requirements.txt

          # Install the application itself
          pip --cache-dir=.pip install -e .
        '';

        installPhase = ''
          # Create wrapper script
          mkdir -p $out/bin

          makeWrapper $out/venv/bin/python $out/bin/rag-client \
            --set LD_LIBRARY_PATH "${libPath}" \
            --set DYLD_LIBRARY_PATH "${libPath}" \
            --add-flags "-m" \
            --add-flags "main"
        '';

        # Skip checks as we're using pip
        doCheck = false;
      };

      apps.default = {
        type = "app";
        program = "${self.packages.${system}.default}/bin/rag-client";
      };

      devShells.default = with pkgs; mkShell {
        buildInputs = [ pythonPackage ] ++ sysLibs ++ [
          libpq.pg_config
        ];

        shellHook = ''
          # Create and activate virtual environment
          [ ! -d .venv ] && python -m venv .venv
          source .venv/bin/activate

          # Set environment
          export PATH=${libpq.pg_config}/bin:$PATH
          export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH
          export DYLD_LIBRARY_PATH=${libPath}:$DYLD_LIBRARY_PATH
          export PYTHONPATH="$PWD:$PYTHONPATH"

          # Update pip and install dependencies
          pip install -U pip
          [ -f requirements.txt ] && pip install -r requirements.txt
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
