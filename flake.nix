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
        '';

        installPhase = ''
          mkdir -p $out/bin
          mkdir -p $out/share/lib/rag-client

          cp -p *.py $out/share/lib/rag-client
          cp -p requirements.txt $out/share/lib/rag-client

          makeWrapper ${pkgs.uv}/bin/uv $out/bin/rag-client \
            --set LD_LIBRARY_PATH "${libPath}" \
            --set DYLD_LIBRARY_PATH "${libPath}" \
            --set PATH "${pkgs.libpq.pg_config}/bin:${pkgs.cmake}/bin:${pkgs.pkg-config}/bin:${pkgs.git}/bin:$PATH"        \
            --set NIX_SSL_CERT_FILE "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt" \
            --set SSL_CERT_FILE "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"     \
            --add-flags "run" \
            --add-flags "--no-config" \
            --add-flags "--with-requirements" \
            --add-flags "$out/share/lib/rag-client/requirements.txt" \
            --add-flags "$out/share/lib/rag-client/main.py"
        '';

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
          # Set environment
          export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH
          export DYLD_LIBRARY_PATH=${libPath}:$DYLD_LIBRARY_PATH
          export PYTHONPATH="$PWD:$PYTHONPATH"

          # Create virtual environment if needed
          if [ ! -d .venv ]; then
            echo "Creating virtual environment..."
            python -m venv .venv
          fi

          # Activate virtual environment
          source .venv/bin/activate

          # Install/sync dependencies only if needed
          # Check if .venv needs update (requirements.lock is newer or .venv is missing marker)
          if [ ! -f .venv/.synced ] || [ requirements.lock -nt .venv/.synced ]; then
            echo "Syncing dependencies with uv..."
            uv pip sync requirements.lock
            touch .venv/.synced
          fi
        '';

        nativeBuildInputs = [
          uv                    # Fast Python package installer
          black                 # Python code formatter
          basedpyright          # LSP server for Python
          isort                 # Sorts imports
          autoflake             # Removes unused imports
          pylint
          ruff
          cmake
          pkg-config
          git
          libpq.pg_config
        ];
      };
    });
}
