{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-21.11";
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

  outputs = {self, nixpkgs, pypi-deps-db, mach-nix }@inp:
    let
      requirements = builtins.readFile ./requirements.txt;
      python = "python310";

      supportedSystems = [ "aarch64-linux" "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;}));
    in
    {
      packages = forAllSystems (system: pkgs: {
        default = mach-nix.lib."${system}".mkPython {
          inherit requirements;
          inherit python;
        };
      });

      apps = forAllSystems (system: pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default.out}/bin/python";
        };
      });
    };
}