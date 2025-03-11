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
    in {
      devShell = pkgs.mkShell {
        nativeBuildInputs = [
          (pkgs.python312.withPackages (
            python-pkgs: with python-pkgs; [
              venvShellHook
              numpy_2
              requests
              stdenv

              fastembed
              qdrant-client
              llama-index-core
              llama-index-embeddings-huggingface
              llama-index-llms-ollama
              llama-index-readers-file
              llama-parse
              nltk
              orgparse
            ]
          ))
          pkgs.black          # Python code formatter
          pkgs.pyright        # LSP server for Python
        ];
      };
    });
}
