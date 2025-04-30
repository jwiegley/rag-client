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
      pkgs = import nixpkgs {
        inherit system;
      };
      pythonEnv = pkgs.python312.withPackages (
        python-pkgs: with python-pkgs; [
          venvShellHook
          numpy_2
          stdenv
          llama-index-core
          llama-index-embeddings-huggingface
          llama-index-readers-file
          llama-parse
          nltk
          orgparse
          pypdf
        ]
      );
    in {
      packages.default = pkgs.stdenv.mkDerivation {
        name = "rag-client";
        src = ./.;
        buildInputs = [ pythonEnv ];
        installPhase = ''
          mkdir -p $out/bin
          cp rag.py $out/bin/rag-client.py
          # Create a wrapper to launch the script with the correct Python
          echo '#!${pythonEnv}/bin/python' > $out/bin/rag-client
          cat rag.py >> $out/bin/rag-client
          chmod +x $out/bin/rag-client
        '';
      };

      devShell = pkgs.mkShell {
        nativeBuildInputs = [
          pythonEnv
          pkgs.black          # Python code formatter
          pkgs.pyright        # LSP server for Python
        ];
      };
    });
}
