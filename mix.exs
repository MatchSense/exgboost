defmodule EXGBoost.MixProject do
  use Mix.Project

  @version "0.11.0"

  def project do
    [
      app: :exgboost,
      version: @version,
      make_precompiler: {:nif, EXGBoost.Precompiler},
      make_precompiler_url:
        "https://github.com/iperks/exgboost/releases/download/#{@version}/@{artefact_filename}",
      make_precompiler_priv_paths: ["libexgboost.*", "lib"],
      # NIF Versions correspond to OTP Releases
      # https://github.com/erlang/otp/blob/d3aa6c044c3927f011fb76ac087d5ce0e814954c/erts/emulator/beam/erl_nif.h#L57
      make_precompiler_nif_versions: [
        versions: ["2.17", "2.18"]
      ],
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      deps: deps(),
      name: "EXGBoost",
      source_url: "https://github.com/iperks/exgboost",
      homepage_url: "https://github.com/iperks/exgboost",
      docs: docs(),
      package: package(),
      before_closing_body_tag: &before_closing_body_tag/1,
      name: "EXGBoost",
      description:
        "Elixir bindings for the XGBoost library. `EXGBoost` provides an implementation of XGBoost that works with
      [Nx](https://hexdocs.pm/nx/Nx.html) tensors. Maintained fork of acalejos/exgboost."
    ]
  end

  def cli do
    [preferred_envs: [docs: :docs, "hex.publish": :docs]]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {EXGBoost.Application, []}
    ]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.4", runtime: false},
      {:nimble_options, "~> 1.0"},
      {:nx, "~> 0.9"},
      {:jason, "~> 1.3"},
      {:ex_doc, "~> 0.40", only: :docs},
      {:cc_precompiler, "~> 0.1.0", runtime: false},
      {:exterval, "0.2.0"},
      {:ex_json_schema, "~> 0.11.4"},
      {:vega_lite, "~> 0.1"},
      {:vega_lite_convert, "~> 1.0.1"},
      {:scidata, "~> 0.1", only: :dev},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["Ian Perks"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/iperks/exgboost"},
      files: [
        "lib",
        "mix.exs",
        "c",
        "Makefile",
        "README.md",
        "LICENSE",
        ".formatter.exs",
        "checksum.exs"
      ]
    ]
  end

  defp docs do
    [
      main: "EXGBoost",
      extras: [
        "notebooks/compiled_benchmarks.livemd",
        "notebooks/iris_classification.livemd",
        "notebooks/quantile_prediction_interval.livemd",
        "notebooks/plotting.livemd"
      ],
      groups_for_extras: [
        Notebooks: Path.wildcard("notebooks/*.livemd")
      ],
      groups_for_functions: [
        "System / Native Config": &(&1[:type] == :system),
        "Training & Prediction": &(&1[:type] == :train_pred),
        Serialization: &(&1[:type] == :serialization),
        Plotting: &(&1[:type] == :plotting)
      ],
      groups_for_modules: [
        Plotting: [EXGBoost.Plotting, EXGBoost.Plotting.Styles],
        Training: [
          EXGBoost.Training,
          EXGBoost.Training.Callback,
          EXGBoost.Booster,
          EXGBoost.Parameters
        ]
      ],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <!-- Render math with KaTeX -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>

    <!-- Render diagrams with Mermaid -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@8.13.3/dist/mermaid.min.js"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        mermaid.initialize({ startOnLoad: false });
        let id = 0;
        for (const codeEl of document.querySelectorAll("pre code.mermaid")) {
          const preEl = codeEl.parentElement;
          const graphDefinition = codeEl.textContent;
          const graphEl = document.createElement("div");
          const graphId = "mermaid-graph-" + id++;
          mermaid.render(graphId, graphDefinition, function (svgSource, bindListeners) {
            graphEl.innerHTML = svgSource;
            bindListeners && bindListeners(graphEl);
            preEl.insertAdjacentElement("afterend", graphEl);
            preEl.remove();
          });
        }
      });
    </script>

    <!-- Render Vega-Lite charts -->
      <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.1"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.18.2"></script>
      <style>
        .vega-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); /* Create as many columns as can fit items of at least 200px */
          column-gap: 200px; /* Add a gap between the grid items */
        }

        .vega-item {
          width: 100%; /* Make the items take up the full width of the grid cell */
        }
      </style>
      <script>
        document.addEventListener("DOMContentLoaded", function () {
          for (const codeEl of document.querySelectorAll("pre code.vega-lite")) {
            try {
              const preEl = codeEl.parentElement;
              const spec = JSON.parse(codeEl.textContent);
              const plotEl = document.createElement("div");
              preEl.insertAdjacentElement("afterend", plotEl);
              vegaEmbed(plotEl, spec);
              preEl.remove();
            } catch (error) {
              console.log("Failed to render Vega-Lite plot: " + error)
            }
          }
        });
      </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end

defmodule EXGBoost.Precompiler do
  @moduledoc """
  A custom CC precompiler for EXGBoost NIF library.
  """

  # Variant-aware artifact naming follows Evision's CPU/CUDA packaging pattern
  # without copying its implementation. The callbacks implement the elixir_make
  # precompiler behaviour documented at:
  # https://elixir-make.hexdocs.pm/precompilation_guide.html#precompiler-module-developer
  @cpu_variant "cpu"
  @broad_cuda_variant "cuda80_86_89_90"
  @cuda_variants %{
    "cuda80" => "80",
    "cuda86" => "86",
    "cuda89" => "89",
    "cuda90" => "90",
    @broad_cuda_variant => "80;86;89;90"
  }
  @variants [@cpu_variant | Map.keys(@cuda_variants)]
  @cuda_fetch_targets ["x86_64-linux-gnu"]

  def current_target do
    with {:ok, target} <- CCPrecompiler.current_target() do
      {:ok, variant_target(target, selected_variant!())}
    end
  end

  def all_supported_targets(:compile) do
    variant = selected_variant!()

    CCPrecompiler.all_supported_targets(:compile)
    |> Enum.filter(&target_supports_variant?(&1, variant))
    |> Enum.map(&variant_target(&1, variant))
  end

  def all_supported_targets(:fetch) do
    base_targets = CCPrecompiler.all_supported_targets(:fetch)

    cpu_targets = Enum.map(base_targets, &variant_target(&1, @cpu_variant))

    cuda_targets =
      for target <- base_targets,
          target in @cuda_fetch_targets,
          variant <- Map.keys(@cuda_variants) do
        variant_target(target, variant)
      end

    cpu_targets ++ cuda_targets
  end

  def build_native(args) do
    with_variant_env(selected_variant!(), fn ->
      ElixirMake.Precompiler.mix_compile(args)
    end)
  end

  def precompile(args, target) do
    {base_target, variant} = split_variant_target!(target)

    with_variant_env(variant, fn ->
      CCPrecompiler.precompile(args, base_target)
    end)
  end

  def post_precompile_target(target) do
    {base_target, _variant} = split_variant_target!(target)
    CCPrecompiler.post_precompile_target(base_target)
  end

  def unavailable_target(_target), do: :compile

  defp selected_variant! do
    explicit_target = System.get_env("EXGBOOST_TARGET")

    variant =
      cond do
        explicit_target not in [nil, ""] ->
          explicit_target

        System.get_env("USE_CUDA") == "ON" ->
          cuda_variant_from_architectures(System.get_env("CUDA_ARCHITECTURES"))

        true ->
          @cpu_variant
      end

    validate_variant!(variant)
  end

  defp cuda_variant_from_architectures(nil), do: @broad_cuda_variant
  defp cuda_variant_from_architectures(""), do: @broad_cuda_variant

  defp cuda_variant_from_architectures(architectures) do
    variant = "cuda" <> String.replace(architectures, ";", "_")

    if Map.has_key?(@cuda_variants, variant) do
      variant
    else
      @broad_cuda_variant
    end
  end

  defp validate_variant!(variant) when variant in @variants, do: variant

  defp validate_variant!(variant) do
    Mix.raise(
      "EXGBOOST_TARGET must be one of #{Enum.join(@variants, ", ")}; got #{inspect(variant)}"
    )
  end

  defp target_supports_variant?(_target, @cpu_variant), do: true
  defp target_supports_variant?(target, _cuda_variant), do: target in @cuda_fetch_targets

  defp variant_target(target, variant), do: target <> "-" <> variant

  defp split_variant_target!(target) do
    case Enum.find(@variants, &String.ends_with?(target, "-" <> &1)) do
      nil ->
        Mix.raise("cannot determine EXGBoost precompiled variant from target #{inspect(target)}")

      variant ->
        base_target = String.replace_suffix(target, "-" <> variant, "")
        {base_target, variant}
    end
  end

  defp with_variant_env(variant, fun) when is_function(fun, 0) do
    saved_env =
      for name <- ["USE_CUDA", "USE_NCCL", "CUDA_ARCHITECTURES"], into: %{} do
        {name, System.get_env(name)}
      end

    apply_variant_env(variant)

    try do
      fun.()
    after
      restore_env(saved_env)
    end
  end

  defp apply_variant_env(@cpu_variant) do
    System.put_env("USE_CUDA", "OFF")
    System.put_env("USE_NCCL", System.get_env("USE_NCCL") || "OFF")
    System.delete_env("CUDA_ARCHITECTURES")
  end

  defp apply_variant_env(variant) do
    System.put_env("USE_CUDA", "ON")
    System.put_env("USE_NCCL", System.get_env("USE_NCCL") || "OFF")
    System.put_env("CUDA_ARCHITECTURES", Map.fetch!(@cuda_variants, variant))
  end

  defp restore_env(saved_env) do
    for {name, value} <- saved_env do
      if is_nil(value) do
        System.delete_env(name)
      else
        System.put_env(name, value)
      end
    end
  end
end
