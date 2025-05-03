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

      llama-index-embeddings-openai-like =
        with pkgs.python312Packages; buildPythonPackage rec {
          pname = "llama-index-embeddings-openai-like";
          version = "0.1.0";
          pyproject = true;

          disabled = pythonOlder "3.8";

          src = fetchPypi {
            pname = "llama_index_embeddings_openai_like";
            inherit version;
            hash = "sha256-4LjKt0lgwu29PZ2/+T/Vh/4Fi1OPZQH0YDQF0Pd7l14=";
          };

          pythonRemoveDeps = [];

          build-system = [ poetry-core ];

          dependencies = [
            llama-index-core
            llama-index-embeddings-openai
          ];

          # Tests are only available in the mono repo
          doCheck = false;

          pythonImportsCheck = [ "llama_index.embeddings.openai_like" ];

          meta = with lib; {
            description = "LlamaIndex Embeddings Integration for OpenAI-Like";
            homepage = "https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms/llama-index-embeddings-openai-like";
            license = licenses.mit;
            maintainers = with maintainers; [ jwiegley ];
          };
        };

      typed-argparse =
        with pkgs.python312Packages; buildPythonPackage rec {
          pname = "typed-argparse";
          version = "0.3.1";
          pyproject = true;

          disabled = pythonOlder "3.8";

          src = fetchPypi {
            pname = "typed-argparse";
            inherit version;
            hash = "sha256-OqxhyqUCBuCA0JoAw/5VK8TmQnOb6u+J9fjBExtdWv4=";
          };

          pythonRemoveDeps = [];

          build-system = [ setuptools ];

          dependencies = [
            typing-extensions
          ];

          # Tests are only available in the mono repo
          doCheck = false;

          pythonImportsCheck = [ "typed_argparse" ];

          meta = with lib; {
            description = "Fully typed argument parsing";
            homepage = "https://pypi.org/project/typed-argparse/";
            license = licenses.mit;
            maintainers = with maintainers; [ jwiegley ];
          };
        };

      pythonEnv = pkgs.python312.withPackages (
        python-pkgs: with python-pkgs; [
          stdenv
          venvShellHook
          llama-index-core
          llama-index-embeddings-gemini
          llama-index-embeddings-google
          llama-index-embeddings-huggingface
          llama-index-embeddings-ollama
          llama-index-embeddings-openai
          llama-index-embeddings-openai-like
          llama-index-llms-llama-cpp
          llama-index-llms-ollama
          llama-index-llms-openai
          llama-index-llms-openai-like
          llama-index-readers-file
          llama-index-vector-stores-postgres
          llama-parse
          typed-argparse
          nltk
          numpy_2
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

      devShell = with pkgs; mkShell {
        nativeBuildInputs = [
          pythonEnv
          black                 # Python code formatter
          # pyright               # LSP server for Python
          basedpyright          # LSP server for Python
          isort                 # Manage imports
        ];
      };
    });
}
