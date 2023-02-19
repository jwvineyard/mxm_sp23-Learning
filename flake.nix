{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-21.11";
    flake-utils.url = "github:numtide/flake-utils";
    pypi-deps-db = {
      url = "github:DavHau/pypi-deps-db";
      flake = false;
    };
    mach-nix = {
      url = "github:DavHau/mach-nix/master";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {self, nixpkgs, flake-utils, pypi-deps-db, mach-nix, }@inp:
    flake-utils.lib.eachDefaultSystem (system: let
      requirements = builtins.readFile ./requirements.txt;
      python = "python310";

      pyEnv = mach-nix.lib."${system}".mkPython {
        inherit requirements;
        inherit python;
      };
    in
    {
      packages = {
        default = pyEnv;
        jupyter = mach-nix.nixpkgs.mkShell {
          buildInputs = [
            pyEnv
          ];

          shellHook = ''
            jupyter lab --notebook-dir=~/
          '';
        };
      };

      apps = {
        default = {
          type = "app";
          program = "${self.packages.${system}.default.out}/bin/python";
        };
      };
    });
}