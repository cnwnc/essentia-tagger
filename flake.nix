# essentia-tagger: MAEST/Discogs-519 classifier service + tag index pipeline.
#
# One flake, role-named programs (server / sync / build / serve / extract).
# Which host runs what is a deployment decision (nixosModules args), never code.
#
# Models are fetched from essentia.upf.edu into the nix store (never vendored):
#   ESSENTIA_EMBEDDING_MODEL   discogs-maest-30s-pw-519l-2.pb     (MAEST embeddings)
#   ESSENTIA_CLASSIFIER_HEAD   genre_discogs519-...-519l-1.pb     (519-style head)
#   ESSENTIA_CLASSES_JSON      genre_discogs519-...-519l-1.json   (class names; builder needs this too)
#
# Packaging note: nixpkgs has no essentia(-tensorflow) and its python tensorflow
# was removed long ago; essentia's prebuilt wheels are cp310-only. => pinned
# nixos-25.05 python310 + a venv built once from the exactly-pinned
# requirements.txt (tensorflow[and-cuda] ships the CUDA libs pip-TF dlopens;
# essentia-tensorflow bundles a monolithic TF 2.5 whose CUDA 11 sonames come
# from the pinned nvidia-*-cu11 wheels; driver libcuda comes from
# /run/opengl-driver). Delete .venv (or the StateDirectory venv) to rebuild.
{
  description = "essentia-tagger: MAEST/Discogs-519 classifier + tag index";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python310;

      essentia-models = pkgs.runCommand "essentia-models"
        {
          embPb = pkgs.fetchurl {
            url = "https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.pb";
            hash = "sha256-kng/6yEYdEPQWLTxbXp29HiI1D+9x6KOi8yOAkYDvSA=";
          };
          embJson = pkgs.fetchurl {
            url = "https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.json";
            hash = "sha256-gyQKpVP/tJGw7FokVl62ElU+XzjaUgdAPCW4kMWzSs0=";
          };
          headPb = pkgs.fetchurl {
            url = "https://essentia.upf.edu/models/classification-heads/genre_discogs519/genre_discogs519-discogs-maest-30s-pw-519l-1.pb";
            hash = "sha256-D11h2bYuSifawFiSbphutCTcoP25IMBmq1MVginP9Jg=";
          };
          headJson = pkgs.fetchurl {
            url = "https://essentia.upf.edu/models/classification-heads/genre_discogs519/genre_discogs519-discogs-maest-30s-pw-519l-1.json";
            hash = "sha256-BwFaifGg6bfNzrY5M3gwI9hfOsSzbOXBtUiL0futIwQ=";
          };
        } ''
        mkdir -p $out
        ln -s $embPb   $out/discogs-maest-30s-pw-519l-2.pb
        ln -s $embJson $out/discogs-maest-30s-pw-519l-2.json
        ln -s $headPb  $out/genre_discogs519-discogs-maest-30s-pw-519l-1.pb
        ln -s $headJson $out/genre_discogs519-discogs-maest-30s-pw-519l-1.json
      '';

      modelEnvAttrs = {
        ESSENTIA_EMBEDDING_MODEL = "${essentia-models}/discogs-maest-30s-pw-519l-2.pb";
        ESSENTIA_CLASSIFIER_HEAD = "${essentia-models}/genre_discogs519-discogs-maest-30s-pw-519l-1.pb";
        ESSENTIA_CLASSES_JSON = "${essentia-models}/genre_discogs519-discogs-maest-30s-pw-519l-1.json";
      };
      modelExports = lib.concatStringsSep "\n"
        (lib.mapAttrsToList (k: v: "export ${k}='${v}'") modelEnvAttrs);

      # Shared venv bootstrap: dev shell and service wrappers use the same logic.
      # Venv location: $TAGGER_VENV, defaulting to $PWD/.venv (dev); systemd units
      # set TAGGER_VENV to their StateDirectory.
      ensureVenv = ''
        export VIRTUAL_ENV="''${TAGGER_VENV:-$PWD/.venv}"
        export TF_CPP_MIN_LOG_LEVEL=3
        export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
        if [ ! -x "$VIRTUAL_ENV/bin/python" ] || [ ! -f "$VIRTUAL_ENV/.installed" ]; then
          echo "essentia-tagger: creating venv + installing pinned requirements (~3.5 GB)..."
          ${python}/bin/python -m venv "$VIRTUAL_ENV"
          "$VIRTUAL_ENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip
          "$VIRTUAL_ENV/bin/python" -m pip install --disable-pip-version-check -r "${./requirements.txt}"
          touch "$VIRTUAL_ENV/.installed"
        fi
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        # libstdc++ from nix (pip wheels), NVIDIA driver shim, CUDA libs from wheels
        _nvidia_libs=""
        for _d in "$VIRTUAL_ENV"/lib/python3.10/site-packages/nvidia/*/lib; do
          [ -d "$_d" ] && _nvidia_libs="$_nvidia_libs$_d:"
        done
        export LD_LIBRARY_PATH="''${_nvidia_libs}${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.zstd}/lib:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        unset _nvidia_libs _d
      '';

      # Role-named app wrapper: ensure env, then exec the program.
      mkWrapper = name: module:
        pkgs.writeShellScriptBin name ''
          set -euo pipefail
          ${ensureVenv}
          ${modelExports}
          exec python "${./src}/${module}" "$@"
        '';
      appOf = name: drv: {
        type = "app";
        program = "${drv}/bin/${name}";
      };
      wrappers = {
        tagger-server = mkWrapper "tagger-server" "server.py";
        tagger-sync = mkWrapper "tagger-sync" "sync.py";
        tagger-build = mkWrapper "tagger-build" "build.py";
        tagger-serve = mkWrapper "tagger-serve" "serve.py";
        tagger-extract = mkWrapper "tagger-extract" "extract.py";
      };
    in
    {
      packages.${system} = wrappers // { inherit essentia-models; };
      apps.${system} = builtins.mapAttrs appOf wrappers;

      devShells.${system}.default = pkgs.mkShell {
        packages = [ python pkgs.jq pkgs.ffmpeg essentia-models ];
        shellHook = ''
          ${ensureVenv}
          ${modelExports}
          echo "essentia-tagger dev shell (venv: $VIRTUAL_ENV)"
        '';
      };

      # Skeleton deployment modules — refine once the programs land.
      # classifier = the GPU box role; index = sync timer + build + blob serve.
      nixosModules.classifier = { config, lib, ... }:
        let cfg = config.services.tagger-classifier;
        in {
          options.services.tagger-classifier = {
            enable = lib.mkEnableOption "tagger classifier server (POST /classify)";
            bind = lib.mkOption {
              type = lib.types.str;
              default = "0.0.0.0";
              description = "Bind address; set to the wireguard IP in prod (wg is the trust boundary).";
            };
            port = lib.mkOption { type = lib.types.port; default = 9478; };
            stateDir = lib.mkOption { type = lib.types.path; default = "/var/lib/tagger-classifier"; };
          };
          config = lib.mkIf cfg.enable {
            systemd.services.tagger-classifier = {
              description = "tagger classifier server (MAEST/Discogs-519)";
              wantedBy = [ "multi-user.target" ];
              environment = modelEnvAttrs // { TAGGER_VENV = "${cfg.stateDir}/venv"; };
              serviceConfig = {
                ExecStart = "${wrappers.tagger-server}/bin/tagger-server --bind ${cfg.bind} --port ${toString cfg.port}";
                WorkingDirectory = cfg.stateDir;
                StateDirectory = "tagger-classifier";
                Restart = "on-failure";
              };
            };
          };
        };

      nixosModules.index = { config, lib, pkgs, ... }:
        let
          cfg = config.services.tagger-index;
          buildHook = pkgs.writeShellScript "tagger-build-hook" ''
            set -euo pipefail
            export TAGGER_VENV="${cfg.stateDir}/venv"
            export ESSENTIA_CLASSES_JSON='${modelEnvAttrs.ESSENTIA_CLASSES_JSON}'
            ${wrappers.tagger-build}/bin/tagger-build --tree ${cfg.tree} --out ${cfg.blob}
          '';
        in {
          options.services.tagger-index = {
            enable = lib.mkEnableOption "tagger index (sync timer + blob build + serve)";
            classifierUrl = lib.mkOption {
              type = lib.types.str;
              example = "http://10.100.1.1:9478";
              description = "Upstream POST /classify endpoint (wg-only).";
            };
            musicDir = lib.mkOption { type = lib.types.path; };
            tree = lib.mkOption { type = lib.types.path; description = "Classifier JSON tree."; };
            blob = lib.mkOption { type = lib.types.path; description = "Output blob path."; };
            bind = lib.mkOption { type = lib.types.str; default = "0.0.0.0"; };
            port = lib.mkOption { type = lib.types.port; default = 9479; };
            stateDir = lib.mkOption { type = lib.types.path; default = "/var/lib/tagger-index"; };
            schedule = lib.mkOption {
              type = lib.types.str;
              default = "*-*-* 03:00:00";
              description = "sync timer calendar spec (sleep-hours window).";
            };
          };
          config = lib.mkIf cfg.enable {
            systemd.services.tagger-sync = {
              description = "tagger sync (diff library, push to classifier, store results)";
              # only the classes json: the .pb models stay on the classifier host
              environment = {
                ESSENTIA_CLASSES_JSON = modelEnvAttrs.ESSENTIA_CLASSES_JSON;
                TAGGER_VENV = "${cfg.stateDir}/venv";
              };
              serviceConfig = {
                Type = "oneshot";
                ExecStart = "${wrappers.tagger-sync}/bin/tagger-sync --music ${cfg.musicDir} --tree ${cfg.tree} --url ${cfg.classifierUrl}";
                WorkingDirectory = cfg.stateDir;
                StateDirectory = "tagger-index";
              };
            };
            systemd.services.tagger-build = {
              description = "tagger blob build";
              environment = { TAGGER_VENV = "${cfg.stateDir}/venv"; };
              serviceConfig = { Type = "oneshot"; ExecStart = "${buildHook}"; };
            };
            systemd.paths.tagger-build = {
              wantedBy = [ "tagger-sync.service" ];
              pathChanged = [ "${cfg.tree}" ];
            };
            systemd.services.tagger-serve = {
              description = "tagger blob server (ETag/Last-Modified, wg-only)";
              environment = { TAGGER_VENV = "${cfg.stateDir}/venv"; };
              wantedBy = [ "multi-user.target" ];
              serviceConfig = {
                ExecStart = "${wrappers.tagger-serve}/bin/tagger-serve --blob ${cfg.blob} --bind ${cfg.bind} --port ${toString cfg.port}";
                WorkingDirectory = cfg.stateDir;
                StateDirectory = "tagger-index";
                Restart = "on-failure";
              };
            };
            systemd.timers.tagger-sync = {
              wantedBy = [ "timers.target" ];
              timerConfig = { OnCalendar = cfg.schedule; Persistent = true; };
            };
          };
        };
    };
}
