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
# Two venv flavors, both built once from exactly-pinned requirement files:
#   full (requirements.txt): essentia-tensorflow + tensorflow[and-cuda] + CUDA
#     wheels (~3.5GB) — server + extract (the GPU side) only.
#   lite (requirements-lite.txt): numpy only — sync + build + serve (the index
#     side never touches a model, it just moves/stores/derives JSON).
#
# essentia packaging reality (why the venv exists at all): nixpkgs has no
# essentia(-tensorflow) and its python tensorflow was removed long ago;
# essentia's prebuilt wheels are cp310-only. => pinned nixos-25.05 python310.
# essentia-tensorflow bundles a monolithic TF 2.5 whose CUDA 11 sonames come
# from the pinned nvidia-*-cu11 wheels; driver libcuda comes from
# /run/opengl-driver. Delete the venv dir to force a rebuild.
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

      # Shared venv bootstrap: dev shell and service wrappers use the same logic.
      # Venv location: $TAGGER_VENV, defaulting to $PWD/.venv (dev); systemd units
      # set TAGGER_VENV to their StateDirectory. `lite` = numpy-only venv for the
      # index side (no models ever load there).
      ensureVenv = lite:
        let
          req = if lite then ./requirements-lite.txt else ./requirements.txt;
          probe = if lite then "import numpy" else "import numpy, essentia.standard";
          nvidiaGlob = lib.optionalString (!lite) ''
            _nv() {
              _n=""
              for _d in "$VIRTUAL_ENV"/lib/python3.10/site-packages/nvidia/*/lib; do
                [ -d "$_d" ] && _n="$_n$_d:"
              done
              export LD_LIBRARY_PATH="''${_n}/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            }
            _nv
          '';
        in ''
          export VIRTUAL_ENV="''${TAGGER_VENV:-$PWD/.venv}"
          export TF_CPP_MIN_LOG_LEVEL=3
          export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          export PATH="${pkgs.zstd}/bin:${pkgs.ffmpeg}/bin:$PATH"
          # pip wheels dlopen libz/libstdc++ — needed by the import probes below
          # too, so this goes BEFORE any venv check (numpy fails on libz otherwise)
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.zstd}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          ${nvidiaGlob}
          _req_hash='${req}'   # store path doubles as a content hash
          _need=0
          { [ ! -x "$VIRTUAL_ENV/bin/python" ] || [ ! -f "$VIRTUAL_ENV/.installed" ] \
            || [ ! -f "$VIRTUAL_ENV/.req-hash" ] \
            || [ "$(cat "$VIRTUAL_ENV/.req-hash" 2>/dev/null)" != "$_req_hash" ]; } && _need=1
          if [ "$_need" = 1 ]; then
            # serialize concurrent bootstraps (dev shells / parallel services)
            exec 9>"$VIRTUAL_ENV.lock"
            flock -w 1800 9 || true
            { [ -x "$VIRTUAL_ENV/bin/python" ] && [ -f "$VIRTUAL_ENV/.installed" ] \
              && [ -f "$VIRTUAL_ENV/.req-hash" ] \
              && [ "$(cat "$VIRTUAL_ENV/.req-hash" 2>/dev/null)" = "$_req_hash" ]; } && _need=0
            if [ "$_need" = 1 ]; then
              # adopt a legacy (pre-hash-marker) venv only if it actually imports
              if [ -f "$VIRTUAL_ENV/.installed" ] && [ ! -f "$VIRTUAL_ENV/.req-hash" ] \
                 && "$VIRTUAL_ENV/bin/python" -c '${probe}' >/dev/null 2>&1; then
                echo "essentia-tagger: adopting existing venv"
                printf '%s' "$_req_hash" > "$VIRTUAL_ENV/.req-hash"
              else
                echo "essentia-tagger: creating venv + installing pinned requirements..."
                rm -rf "$VIRTUAL_ENV"
                ${python}/bin/python -m venv "$VIRTUAL_ENV"
                "$VIRTUAL_ENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip
                "$VIRTUAL_ENV/bin/python" -m pip install --disable-pip-version-check -r '${req}'
                ${lib.optionalString (!lite) "_nv"}
                # hard verify: the import MUST succeed before the venv is trusted
                # (a sentinel over a broken venv caused the PASSENGER crash loop)
                if ! "$VIRTUAL_ENV/bin/python" -c '${probe}'; then
                  echo "essentia-tagger: FATAL: venv installed but import probe failed" >&2
                  exit 1
                fi
                touch "$VIRTUAL_ENV/.installed"
                printf '%s' "$_req_hash" > "$VIRTUAL_ENV/.req-hash"
              fi
            fi
            exec 9>&-
          fi
          export PATH="$VIRTUAL_ENV/bin:$PATH"
          ${lib.optionalString (!lite) "_nv"}
          # pip wheels dlopen libz/libstdc++: needed on both lite and full
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.zstd}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          ${lib.optionalString (!lite) ''
            # NVIDIA driver shim + CUDA libs from the pip wheels (GPU side only)
            _nvidia_libs=""
            for _d in "$VIRTUAL_ENV"/lib/python3.10/site-packages/nvidia/*/lib; do
              [ -d "$_d" ] && _nvidia_libs="$_nvidia_libs$_d:"
            done
            export LD_LIBRARY_PATH="''${_nvidia_libs}/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            unset _nvidia_libs _d
          ''}
        '';

      # Role-named app wrapper: ensure env, then exec the program.
      # lite = index-side tools (sync/build/serve); everything else is GPU-side.
      mkWrapper = name: module: lite:
        pkgs.writeShellScriptBin name ''
          set -euo pipefail
          ${ensureVenv lite}
          ${lib.concatStringsSep "\n"
            (lib.mapAttrsToList (k: v: "export ${k}='${v}'") modelEnvAttrs)}
          exec python "${./src}/${module}" "$@"
        '';
      appOf = name: drv: { type = "app"; program = "${drv}/bin/${name}"; };
      wrappers = {
        tagger-server = mkWrapper "tagger-server" "server.py" false;
        tagger-extract = mkWrapper "tagger-extract" "extract.py" false;
        tagger-sync = mkWrapper "tagger-sync" "sync.py" true;
        tagger-build = mkWrapper "tagger-build" "build.py" true;
        tagger-atlas = mkWrapper "tagger-atlas" "atlas.py" true;
        tagger-serve = mkWrapper "tagger-serve" "serve.py" true;
        tagger-radiomap = mkWrapper "tagger-radiomap" "radiomap.py" true;
      };
    in
    {
      packages.${system} = wrappers // { inherit essentia-models; };
      apps.${system} = builtins.mapAttrs appOf wrappers;

      # Force-evaluate both modules ENABLED (incl. the navidromeEnvFile branch)
      # so option type errors (e.g. serviceConfig not an attrset) fail here and
      # not in the user's config. nix flake check's shallow module check does
      # not catch those, because mkIf-guarded config never gets evaluated.
      checks.${system} =
        let
          eval = modules: (import "${nixpkgs}/nixos/lib/eval-config.nix" {
            inherit system;
            inherit modules;
          });
          classifierEval = eval [
            self.nixosModules.classifier {
              services.tagger-classifier.enable = true;
              services.tagger-classifier.bind = "127.0.0.1";
            }
          ];
          indexEval = eval [
            self.nixosModules.index {
              services.tagger-index.enable = true;
              services.tagger-index.musicDir = "/data/music";
              services.tagger-index.classifierUrl = "http://10.100.1.1:9478";
              services.tagger-index.navidromeEnvFile = "/etc/navidrome.env";
            }
          ];
        in {
          modules-eval = pkgs.writeText "essentia-tagger-modules-eval.json"
            (builtins.toJSON {
              classifier = {
                inherit (classifierEval.config.systemd.services.tagger-classifier) serviceConfig;
                venv = classifierEval.config.systemd.services.tagger-venv.serviceConfig.ExecStart;
                firewall = classifierEval.config.networking.firewall.allowedTCPPorts;
              };
              index = {
                sync = indexEval.config.systemd.services.tagger-sync.serviceConfig.ExecStart;
                inherit (indexEval.config.systemd.services.tagger-atlas) serviceConfig;
                build = indexEval.config.systemd.services.tagger-build.serviceConfig.ExecStart;
                serve = indexEval.config.systemd.services.tagger-serve.serviceConfig.ExecStart;
                timer = indexEval.config.systemd.timers.tagger-sync.timerConfig.OnCalendar;
              };
            });
        };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ python pkgs.jq pkgs.ffmpeg pkgs.zstd essentia-models ];
        shellHook = ''
          ${ensureVenv false}
          ${lib.concatStringsSep "\n"
            (lib.mapAttrsToList (k: v: "export ${k}='${v}'") modelEnvAttrs)}
          echo "essentia-tagger dev shell (venv: $VIRTUAL_ENV)"
        '';
      };

      # ------------------------------------------------------------------
      # classifier: the GPU host role. Deployment declares only the wg bind
      # address; everything else defaults. DynamicUser keeps it user-free.
      # ------------------------------------------------------------------
      nixosModules.classifier = { config, lib, ... }:
        let cfg = config.services.tagger-classifier;
        in {
          options.services.tagger-classifier = {
            enable = lib.mkEnableOption "tagger classifier server (POST /classify)";
            bind = lib.mkOption {
              type = lib.types.str;
              default = "127.0.0.1";
              description = "Bind address; set to the wireguard IP in prod (wg is the trust boundary, no auth).";
            };
            port = lib.mkOption { type = lib.types.port; default = 9478; };
            idleTimeout = lib.mkOption {
              type = lib.types.ints.positive;
              default = 600;
              description = "Seconds of inactivity before unloading models (auto-warm on next request).";
            };
          };
          config = lib.mkIf cfg.enable {
            users.groups.essentia = { };
            users.users.essentia = {
              isSystemUser = true;
              group = "essentia";
              description = "tagger service user";
            };
            # index host (and anything else on wg) must reach POST /classify
            networking.firewall.allowedTCPPorts = [ cfg.port ];
            systemd.services.tagger-venv = {
              description = "tagger venv bootstrap (full: essentia-tensorflow + CUDA, ~3.5GB, needs network once)";
              serviceConfig = {
                Type = "oneshot";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "tagger-classifier";
                TimeoutStartSec = "2h";
              };
              environment.TAGGER_VENV = "/var/lib/tagger-classifier/venv";
              script = (ensureVenv false) + "\n";
            };
            systemd.services.tagger-classifier = {
              description = "tagger classifier server (MAEST/Discogs-519, auto-warm)";
              wantedBy = [ "multi-user.target" ];
              requires = [ "tagger-venv.service" ];
              after = [ "tagger-venv.service" "network.target" ];
              environment = modelEnvAttrs // {
                TAGGER_VENV = "/var/lib/tagger-classifier/venv";
                # don't let TF grab ~all VRAM at warmup (this is a gaming PC);
                # allocate on demand instead of ~9GB upfront
                TF_FORCE_GPU_ALLOW_GROWTH = "true";
              };
              serviceConfig = {
                ExecStart = "${wrappers.tagger-server}/bin/tagger-server --bind ${cfg.bind} --port ${toString cfg.port} --idle-timeout ${toString cfg.idleTimeout}";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "tagger-classifier";
                SupplementaryGroups = [ "video" "render" ];
                Restart = "on-failure";
                RestartSec = 5;
                # TF/numpy mmap large .so files out of the state dir; explicit
                # hardening-offs — DynamicUser sandboxing broke that with
                # "failed to map segment from shared object" (crash loop)
                PrivateTmp = false;
                PrivateDevices = false;
                MemoryDenyWriteExecute = false;
                ProtectSystem = "full";
                ProtectHome = true;
                NoNewPrivileges = true;
              };
            };
          };
        };

      # ------------------------------------------------------------------
      # index: sync timer + blob build + blob GET endpoint. Deployment
      # declares only classifierUrl, musicDir, and (optionally) the schedule.
      # Runs as system user `essentia` in /var/lib/essentia (created here):
      #   data/  = classifier JSON tree (backfill rsync target)
      #   blob/  = built similarity blob (served on :9478)
      # ------------------------------------------------------------------
      nixosModules.index = { config, lib, pkgs, ... }:
        let
          cfg = config.services.tagger-index;
          tree' = if cfg.tree != null then toString cfg.tree else "${cfg.stateDir}/data";
          blob' = if cfg.blob != null then toString cfg.blob else "${cfg.stateDir}/blob/albums.bin";
          atlasDir' = if cfg.atlasDir != null then toString cfg.atlasDir else "${cfg.stateDir}/blob";
          atlasHook = pkgs.writeShellScript "tagger-atlas-hook" ''
            set -euo pipefail
            export TAGGER_VENV="${cfg.stateDir}/venv"
            export ESSENTIA_CLASSES_JSON='${modelEnvAttrs.ESSENTIA_CLASSES_JSON}'
            # NAVIDROME_* env arrives via the unit's EnvironmentFile (if set)
            ${wrappers.tagger-atlas}/bin/tagger-atlas --tree ${tree'} --music ${cfg.musicDir} --out-dir ${atlasDir'}
          '';
          buildHook = pkgs.writeShellScript "tagger-build-hook" ''
            set -euo pipefail
            export TAGGER_VENV="${cfg.stateDir}/venv"
            export ESSENTIA_CLASSES_JSON='${modelEnvAttrs.ESSENTIA_CLASSES_JSON}'
            MAN=""
            if [ -f "${atlasDir'}/atlas-manifest.json" ]; then
              MAN="--atlas-manifest ${atlasDir'}/atlas-manifest.json"
            fi
            ${wrappers.tagger-build}/bin/tagger-build --tree ${tree'} --out ${blob'} $MAN
            ${wrappers.tagger-radiomap}/bin/tagger-radiomap --blob ${blob'} --music ${cfg.musicDir} --out ${atlasDir'}/radio-map.json || echo "WARN: radio map build failed; /radio-map unavailable" 
          '';
        in {
          options.services.tagger-index = {
            enable = lib.mkEnableOption "tagger index (sync timer + blob build + blob serve)";
            classifierUrl = lib.mkOption {
              type = lib.types.str;
              example = "http://10.100.1.1:9478";
              description = "Upstream POST /classify endpoint (wg-only).";
            };
            musicDir = lib.mkOption {
              type = lib.types.path;
              description = "Music library root (must be readable by the essentia user).";
            };
            schedule = lib.mkOption {
              type = lib.types.str;
              default = "*-*-* 03:30:00";
              description = "systemd OnCalendar spec for the sync timer (sleep-hours window).";
            };
            bind = lib.mkOption { type = lib.types.str; default = "0.0.0.0"; };
            port = lib.mkOption { type = lib.types.port; default = 9478; };
            stateDir = lib.mkOption { type = lib.types.path; default = "/var/lib/essentia"; };
            tree = lib.mkOption {
              type = with lib.types; nullOr path;
              default = null;
              description = "Classifier JSON tree; defaults to \${stateDir}/data (backfill rsync target).";
            };
            blob = lib.mkOption {
              type = with lib.types; nullOr path;
              default = null;
              description = "Built similarity blob; defaults to \${stateDir}/blob/albums.bin (served over HTTP).";
            };
            navidromeEnvFile = lib.mkOption {
              type = with lib.types; nullOr (either str path);
              default = null;
              description = "Env file with NAVIDROME_BASE_URL/USERNAME/PASSWORD (Subsonic auth, unprivileged user) for the atlas builder.";
            };
            atlasDir = lib.mkOption {
              type = with lib.types; nullOr path;
              default = null;
              description = "Atlas output dir; defaults to \${stateDir}/blob (atlas-N.webp + atlas-manifest.json + cache/).";
            };
          };
          config = lib.mkIf cfg.enable {
            users.groups.essentia = { };
            users.users.essentia = {
              isSystemUser = true;
              group = "essentia";
              description = "tagger index service user";
            };
            # the KAZOOIE backend (and any wg peer) must reach GET /blob
            networking.firewall.allowedTCPPorts = [ cfg.port ];
            systemd.services.tagger-venv = {
              description = "tagger venv bootstrap (lite: numpy only)";
              serviceConfig = {
                Type = "oneshot";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "essentia";
              };
              environment.TAGGER_VENV = "${cfg.stateDir}/venv";
              script = (ensureVenv true) + "\n";
            };
            systemd.services.tagger-sync = {
              description = "tagger sync (diff library, push to classifier, store results)";
              after = [ "network.target" "tagger-venv.service" ];
              requires = [ "tagger-venv.service" ];
              environment = {
                ESSENTIA_CLASSES_JSON = modelEnvAttrs.ESSENTIA_CLASSES_JSON;
                TAGGER_VENV = "${cfg.stateDir}/venv";
              };
              serviceConfig = {
                Type = "oneshot";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "essentia";
                ExecStart = "${wrappers.tagger-sync}/bin/tagger-sync --music ${cfg.musicDir} --tree ${tree'} --url ${cfg.classifierUrl}";
                TimeoutStartSec = "8h";
              };
              # chain: sync -> atlas -> build (serve stays resident)
              onSuccess = [ "tagger-atlas.service" ];
            };
            systemd.services.tagger-atlas = {
              description = "tagger atlas build (navidrome covers → webp sprite sheets)";
              after = [ "tagger-venv.service" ];
              requires = [ "tagger-venv.service" ];
              onSuccess = [ "tagger-build.service" ];
              environment = {
                ESSENTIA_CLASSES_JSON = modelEnvAttrs.ESSENTIA_CLASSES_JSON;
                TAGGER_VENV = "${cfg.stateDir}/venv";
              };
              serviceConfig = {
                Type = "oneshot";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "essentia";
                ExecStart = "${atlasHook}";
              } // (lib.optionalAttrs (cfg.navidromeEnvFile != null) {
                EnvironmentFile = toString cfg.navidromeEnvFile;
              });
            };
            systemd.services.tagger-build = {
              description = "tagger blob build (album medians + similarity matrices)";
              after = [ "tagger-venv.service" ];
              requires = [ "tagger-venv.service" ];
              environment = {
                ESSENTIA_CLASSES_JSON = modelEnvAttrs.ESSENTIA_CLASSES_JSON;
                TAGGER_VENV = "${cfg.stateDir}/venv";
              };
              serviceConfig = {
                Type = "oneshot";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "essentia";
                ExecStart = "${buildHook}";
              };
            };
            # chain: sync -> atlas -> build (atlas onSuccess is set in its unit;
            # serve stays resident)
            systemd.services.tagger-serve = {
              description = "tagger blob server (GET blob + atlas, ETag/304; wg-only reachability)";
              wantedBy = [ "multi-user.target" ];
              after = [ "tagger-venv.service" "network.target" ];
              requires = [ "tagger-venv.service" ];
              environment = { TAGGER_VENV = "${cfg.stateDir}/venv"; };
              serviceConfig = {
                ExecStart = "${wrappers.tagger-serve}/bin/tagger-serve --blob ${blob'} --atlas-dir ${atlasDir'} --radio-map ${atlasDir'}/radio-map.json --bind ${cfg.bind} --port ${toString cfg.port}";
                User = "essentia";
                Group = "essentia";
                StateDirectory = "essentia";
                Restart = "on-failure";
                RestartSec = 5;
                NoNewPrivileges = true;
              };
            };
            systemd.timers.tagger-sync = {
              wantedBy = [ "timers.target" ];
              timerConfig = {
                OnCalendar = cfg.schedule;
                Persistent = true;
                RandomizedDelaySec = "10min";
              };
            };
          };
        };
    };
}
