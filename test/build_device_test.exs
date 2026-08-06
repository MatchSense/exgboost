defmodule EXGBoostBuildDeviceTest do
  use ExUnit.Case, async: false

  @device_env "EXGBOOST_TEST_DEVICE"

  test "native XGBoost build can train and predict on the selected device" do
    build_info = EXGBoost.xgboost_build_info()
    device = selected_device(build_info)

    assert_build_supports_device!(build_info, device)

    booster =
      EXGBoost.train(
        training_x(),
        training_y(),
        device: device,
        tree_method: :hist,
        objective: :reg_squarederror,
        num_boost_rounds: 4,
        max_depth: 2,
        seed: 0,
        verbose_eval: false
      )

    predictions =
      EXGBoost.inplace_predict(
        booster,
        prediction_x(),
        validate_features: false
      )

    assert Nx.shape(predictions) == {2}
    assert Enum.all?(Nx.to_flat_list(predictions), &is_number/1)
  end

  test "native XGBoost build info matches the selected device" do
    build_info = EXGBoost.xgboost_build_info()
    device = selected_device(build_info)

    assert is_boolean(Map.fetch!(build_info, "USE_CUDA"))
    assert_build_supports_device!(build_info, device)
  end

  defp selected_device(build_info) do
    case System.get_env(@device_env, "auto") do
      "auto" ->
        if cuda_build?(build_info), do: :cuda, else: :cpu

      "cpu" ->
        :cpu

      "cuda" ->
        :cuda

      "gpu" ->
        :cuda

      other ->
        flunk("#{@device_env} must be auto, cpu, cuda, or gpu; got #{inspect(other)}")
    end
  end

  defp assert_build_supports_device!(build_info, :cuda) do
    assert cuda_build?(build_info),
           "expected a CUDA-enabled XGBoost build; compile with USE_CUDA=ON before running CUDA tests"
  end

  defp assert_build_supports_device!(_build_info, :cpu), do: :ok

  defp cuda_build?(%{"USE_CUDA" => true}), do: true
  defp cuda_build?(_build_info), do: false

  defp training_x do
    Nx.tensor(
      [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
      ],
      type: {:f, 32}
    )
  end

  defp training_y do
    Nx.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0], type: {:f, 32})
  end

  defp prediction_x do
    Nx.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], type: {:f, 32})
  end
end
