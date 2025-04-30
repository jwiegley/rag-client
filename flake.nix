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

      # llama-index-vector-stores-faiss =
      #   with pkgs; python312Packages.buildPythonPackage rec {
      #     pname = "llama-index-vector-stores-faiss";
      #     version = "0.3.0";
      #     pyproject = true;

      #     src = fetchPypi {
      #       pname = "llama_index_vector_stores_faiss";
      #       inherit version;
      #       hash = "sha256-yd+Z3QD+cFhgbvT84RNTX6MLc+3WUBNr6HybWyQN8/k=";
      #     };

      #     build-system = with python312Packages; [ poetry-core ];

      #     dependencies = with python312Packages; [
      #       llama-index-core
      #       faiss
      #     ];

      #     pythonImportsCheck = [ "llama_index.vector_stores.faiss" ];

      #     meta = with lib; {
      #       description = "LlamaIndex Vector Store Integration for FAISS";
      #       homepage = "https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/vector_stores/llama-index-vector-stores-faiss";
      #       license = licenses.mit;
      #       maintainers = with maintainers; [ fab ];
      #     };
      #   };

      pythonEnv = pkgs.python312.withPackages (
        python-pkgs: with python-pkgs; [
          venvShellHook
          numpy_2
          stdenv
          llama-index-core
          llama-index-embeddings-huggingface
          # faiss
          # llama-index-vector-stores-faiss
          llama-index-readers-file
          llama-parse
          nltk
          orgparse
          pypdf
          xdg-base-dirs
        ]
      );
    in {
      packages.default = pkgs.stdenv.mkDerivation {
        name = "rag-client";
        src = ./.;
        buildInputs = [ pythonEnv ];
        installPhase = ''
          mkdir -p $out/bin
          # cp rag-client.py $out/bin
          # Create a wrapper to launch the script with the correct Python
          echo '#!${pythonEnv}/bin/python' > $out/bin/rag-client
          cat rag-client.py >> $out/bin/rag-client
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
