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
          # Create and activate virtual environment
          [ ! -d .venv ] && python -m venv .venv
          source .venv/bin/activate

          # Set environment
          export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH
          export DYLD_LIBRARY_PATH=${libPath}:$DYLD_LIBRARY_PATH
          export PYTHONPATH="$PWD:$PYTHONPATH"

          # Update pip and install dependencies
          pip --cache-dir=.venv/pip install -U pip
          pip --cache-dir=.venv/pip install -r requirements.txt
        '';

        nativeBuildInputs = [
          black                 # Python code formatter
          basedpyright          # LSP server for Python
          isort                 # Sorts imports
          autoflake             # Removes unused imports
          pylint
          cmake
          pkg-config
          git
          libpq.pg_config
        ];
      };
    });
}
