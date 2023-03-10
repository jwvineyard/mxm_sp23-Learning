{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryEnv defaultPoetryOverrides;
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages = {
          default = mkPoetryEnv { 
            projectDir = self; 
            overrides = defaultPoetryOverrides.extend
              (self: super: {
                gymnasium = super.gymnasium.overridePythonAttrs
                (old: { buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ]; });

                gymnasium-notices = super.gymnasium-notices.overridePythonAttrs
                (old: { buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ]; });

                jax-jumpy = super.jax-jumpy.overridePythonAttrs
                (old: { buildInputs = (old.buildInputs or [ ]) ++ [ super.hatchling ]; });
              });
              extraPackages = ps: with ps; [ ps.poetry ];
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [ poetry2nix.packages.${system}.poetry ];
        };

        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
          exePath = "/bin/python";
        };
      });
}
