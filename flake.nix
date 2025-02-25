{
  description = "Python implementation of Graph SLAM";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

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
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

        python = pkgs.python312;
        pythonPackages = pkgs.python312Packages;

        pyprojectOverrides = final: prev: {
          g2o-python = prev.g2o-python.overrideAttrs {
            env = prev.g2o-python.env or { } // {
              GIT_CONFIG_COUNT = 1;
              GIT_CONFIG_KEY_0 = "url.https://github.com/.insteadOf";
              GIT_CONFIG_VALUE_0 = "git@github.com:";
            };

            postPatch = ''
              substituteInPlace g2o/python/CMakeLists.txt \
                --replace-fail "FetchContent_MakeAvailable(pybind11)" \
                "find_package(pybind11 REQUIRED)"
            '';

            preFixup = ''
              echo 'post install'
              for lib in $out/${final.python.sitePackages}/*/*.so; do
                patchelf --add-rpath "$out/lib64" "$lib"
              done
            '';

            dontUseCmakeConfigure = true;

            nativeBuildInputs =
              prev.g2o-python.nativeBuildInputs
              ++ [
                (final.resolveBuildSystem {
                  setuptools = [ ];
                  scikit-build = [ ];
                  cmake = [ ];
                  ninja = [ ];
                })
              ]
              ++ [
                pkgs.spdlog
                pkgs.eigen
                pkgs.suitesparse
                pythonPackages.pybind11
                pkgs.patchelf
              ];
          };

          raylib = prev.raylib.overrideAttrs {
            postFixup = ''
              for lib in $out/${final.python.sitePackages}/*/*.so; do
                patchelf $lib --add-rpath ${
                  pkgs.lib.makeLibraryPath [
                    pkgs.xorg.libX11
                    pkgs.xorg.libXcursor
                    pkgs.xorg.libXi
                    pkgs.xorg.libXrandr
                    pkgs.wayland
                    pkgs.libxkbcommon
                    pkgs.libGL
                  ]
                }
              done
            '';
          };
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
            stdenv = pkgs.stdenv.override {
              targetPlatform = pkgs.stdenv.targetPlatform // {
                darwinSdkVersion = "15.1";
              };
            };
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );

        venv = pythonSet.mkVirtualEnv "pygraph-env" workspace.deps.default;
      in
      {
        packages = {
          default = pkgs.writeShellApplication {
            name = "pygraph";
            text = ''
              exec ${venv}/bin/python -m src.slam "$@"
            '';
          };
        };

        devShells.default =
          let
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              root = "$REPO_ROOT";
              members = [ "pygraphslam" ];
            };

            editablePythonSet = pythonSet.overrideScope (
              pkgs.lib.composeManyExtensions [
                editableOverlay

                (final: prev: {
                  pygraph = prev.pygraph.overrideAttrs (old: {
                    src = pkgs.lib.fileset.toSource {
                      root = old.src;
                      fileset = pkgs.lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/README.md")
                      ];
                    };

                    nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
                  });
                })
              ]
            );

            virtualenv = editablePythonSet.mkVirtualEnv "pygraph-dev-env" workspace.deps.all;
          in
          pkgs.mkShell {
            packages = [
              virtualenv
              pythonPackages.uv
              pythonPackages.ruff
              pythonPackages.python-lsp-server
              pkgs.glfw
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = "${virtualenv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
      }
    );
}
