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
        overlays = [
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (pyFinal: pyPrev: {
                # spacy = pyPrev.spacy.overridePythonAttrs (o: rec {
                #   meta = o.meta // { broken = false; };
                # });

                # banks = with pkgs; with pyPrev; buildPythonPackage rec {
                #   pname = "banks";
                #   version = "2.1.2";
                #   pyproject = true;

                #   src = fetchFromGitHub {
                #     owner = "masci";
                #     repo = "banks";
                #     tag = "v${version}";
                #     hash = "sha256-lOlNYIBMa3G06t5KfRWNd/d8aXjxnWp11n8Kw7Ydy+Y=";
                #   };

                #   SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";

                #   build-system = [ hatchling ];

                #   dependencies = [
                #     deprecated
                #     eval-type-backport
                #     griffe
                #     jinja2
                #     platformdirs
                #     pydantic
                #   ];

                #   optional-dependencies = {
                #     all = [
                #       litellm
                #       redis
                #     ];
                #   };

                #   nativeCheckInputs = [
                #     pytest-asyncio
                #     pytestCheckHook
                #   ] ++ lib.flatten (builtins.attrValues optional-dependencies);

                #   pythonImportsCheck = [ "banks" ];

                #   meta = {
                #     description = "Module that provides tools and functions to build prompts text and chat messages from generic blueprints";
                #     homepage = "https://github.com/masci/banks";
                #     changelog = "https://github.com/masci/banks/releases/tag/${src.tag}";
                #     license = lib.licenses.mit;
                #     maintainers = with lib.maintainers; [ fab ];
                #   };
                # };

                # llama-index-core = pyPrev.llama-index-core.overridePythonAttrs (o: rec {
                #   version = "0.12.34";
                #   src = pkgs.fetchFromGitHub {
                #     owner = "run-llama";
                #     repo = "llama_index";
                #     tag = "v${version}";
                #     hash = "sha256-EC84HBcMXuvhcKGyJu9Fy7nFcHLP+vXDbF4ekhc0LlM=";
                #   };
                #   dependencies = o.dependencies ++ (with pyFinal; [
                #     banks
                #   ]);
                #   nativeBuildInputs = with pyPrev; [
                #     hatchling
                #   ];
                # });
              })
            ];
          })
        ];
      };

      llama-index-llms-llama-cpp =
        with pkgs.python3Packages; buildPythonPackage rec {
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
        with pkgs.python3Packages; buildPythonPackage rec {
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
        with pkgs.python3Packages; buildPythonPackage rec {
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

      pythonEnv = pkgs.python3.withPackages (
        python-pkgs: with python-pkgs; [
          stdenv
          venvShellHook
          llama-index-core
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
      inherit pkgs;

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
          basedpyright          # LSP server for Python
          isort                 # Sorts imports
          autoflake             # Removes unused imports
        ];
      };
    });
}
