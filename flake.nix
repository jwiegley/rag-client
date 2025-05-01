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

      llama-index-llms-llama-cpp =
        with pkgs.python312Packages; buildPythonPackage rec {
          pname = "llama-index-llms-llama-cpp";
          version = "0.4.0";
          pyproject = true;

          disabled = pythonOlder "3.8";

          src = fetchPypi {
            pname = "llama_index_llms_llama_cpp";
            inherit version;
            hash = "sha256-thW8QaoHksFNN56WUIE7hpQ1qrrzhHlP72uOijZ+np0=";
          };

          pythonRemoveDeps = [];

          build-system = [ poetry-core ];

          dependencies = [
            llama-index-core
            llama-cpp-python
          ];

          # Tests are only available in the mono repo
          doCheck = false;

          pythonImportsCheck = [ "llama_index.llms.llama_cpp" ];

          meta = with lib; {
            description = "LlamaIndex LLMS Integration for LlamaCpp";
            homepage = "https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms/llama-index-llms-llama-cpp";
            license = licenses.mit;
            maintainers = with maintainers; [ jwiegley ];
          };
        };

      pythonEnv = pkgs.python312.withPackages (
        python-pkgs: with python-pkgs; [
          venvShellHook
          numpy_2
          stdenv
          llama-index-core
          llama-index-embeddings-huggingface
          llama-index-llms-llama-cpp
          llama-index-vector-stores-postgres
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
