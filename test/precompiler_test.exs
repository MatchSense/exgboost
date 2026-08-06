defmodule EXGBoostPrecompilerTest do
  use ExUnit.Case, async: false

  @target_env "EXGBOOST_TARGET"
  @use_cuda_env "USE_CUDA"
  @cuda_arch_env "CUDA_ARCHITECTURES"

  setup do
    saved_env =
      for name <- [@target_env, @use_cuda_env, @cuda_arch_env], into: %{} do
        {name, System.get_env(name)}
      end

    on_exit(fn -> restore_env(saved_env) end)

    :ok
  end

  test "current target defaults to cpu variant" do
    System.delete_env(@target_env)
    System.delete_env(@use_cuda_env)
    System.delete_env(@cuda_arch_env)

    assert {:ok, target} = EXGBoost.Precompiler.current_target()
    assert String.ends_with?(target, "-cpu")
  end

  test "EXGBOOST_TARGET selects a CUDA artifact variant" do
    System.put_env(@target_env, "cuda89")

    assert {:ok, target} = EXGBoost.Precompiler.current_target()
    assert String.ends_with?(target, "-cuda89")
  end

  test "USE_CUDA and CUDA_ARCHITECTURES select a CUDA artifact variant" do
    System.delete_env(@target_env)
    System.put_env(@use_cuda_env, "ON")
    System.put_env(@cuda_arch_env, "80;86;89;90")

    assert {:ok, target} = EXGBoost.Precompiler.current_target()
    assert String.ends_with?(target, "-cuda80_86_89_90")
  end

  test "fetch targets include CPU targets and Linux CUDA targets" do
    targets = EXGBoost.Precompiler.all_supported_targets(:fetch)

    assert Enum.any?(targets, &String.ends_with?(&1, "-cpu"))
    assert "x86_64-linux-gnu-cuda89" in targets
    assert "x86_64-linux-gnu-cuda80_86_89_90" in targets
  end

  test "invalid EXGBOOST_TARGET fails clearly" do
    System.put_env(@target_env, "cuda77")

    assert_raise Mix.Error, ~r/EXGBOOST_TARGET must be one of/, fn ->
      EXGBoost.Precompiler.current_target()
    end
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
