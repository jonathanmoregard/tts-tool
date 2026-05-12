{
  description = "tts-tool: CleanText (stdin) -> MP3 via Fish Audio TTS.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      mkPyEnv = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
        in
        (pkgs.callPackage pyproject-nix.build.packages { inherit python; })
          .overrideScope (lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
          ]);

      mkWrapped = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = mkPyEnv system;
          venv = pythonSet.mkVirtualEnv "tts-tool-env" workspace.deps.default;
        in
        pkgs.runCommand "tts-tool"
          {
            nativeBuildInputs = [ pkgs.makeWrapper ];
            meta = {
              description = "CleanText -> MP3 via Fish Audio TTS";
              mainProgram = "tts-tool";
            };
          } ''
            mkdir -p $out/bin
            makeWrapper ${venv}/bin/tts-tool $out/bin/tts-tool \
              --prefix PATH : ${lib.makeBinPath [ pkgs.ffmpeg-headless ]}
          '';
    in
    {
      packages = forAllSystems (system: { default = mkWrapped system; });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${mkWrapped system}/bin/tts-tool";
          meta = (mkWrapped system).meta;
        };
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = (mkPyEnv system).overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "tts-tool-dev-env"
            workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [ virtualenv pkgs.uv pkgs.ffmpeg-headless ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = mkPyEnv system;
          virtualenv = pythonSet.mkVirtualEnv "tts-tool-test-env"
            workspace.deps.all;
        in
        {
          pytest = pkgs.runCommand "tts-tool-pytest"
            {
              nativeBuildInputs = [ virtualenv pkgs.ffmpeg-headless ];
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            } ''
              cp -r ${./.}/. .
              chmod -R u+w .
              ${virtualenv}/bin/python -m pytest -q
              touch $out
            '';
        }
      );
    };
}
