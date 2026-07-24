defmodule NifTest do
  use ExUnit.Case, async: true

  import EXGBoost.Internal
  import EXGBoost.ArrayInterface, only: [from_tensor: 1]

  test "exgboost_version" do
    assert EXGBoost.NIF.xgboost_version() |> unwrap!() != :error
  end

  test "unwrap! raises string error for charlist reasons" do
    assert_raise ArgumentError, "boom", fn ->
      unwrap!({:error, ~c"boom"})
    end
  end

  test "build_info" do
    assert EXGBoost.NIF.xgboost_build_info() |> unwrap!() != :error
  end

  test "set_global_config" do
    assert EXGBoost.NIF.set_global_config(~c'{"use_rmm":false,"verbosity":1}') == :ok

    assert EXGBoost.NIF.set_global_config(~c'{"use_rmm":false,"verbosity": true}') ==
             {:error, ~c"Invalid Parameter format for verbosity expect int but value='true'"}
  end

  test "get_global_config" do
    assert EXGBoost.NIF.get_global_config() |> unwrap!() != :error
  end

  test "dmatrix_create_from_uri" do
    config = Jason.encode!(%{uri: "test/data/train.txt?format=libsvm"})
    assert EXGBoost.NIF.dmatrix_create_from_uri(config) |> unwrap!() != :error
  end

  test "dmatrix_create_from_sparse" do
    indptr = Nx.tensor([0, 22])
    ncols = 127

    indices =
      Nx.tensor([
        1,
        9,
        19,
        21,
        24,
        34,
        36,
        39,
        42,
        53,
        56,
        65,
        69,
        77,
        86,
        88,
        92,
        95,
        102,
        106,
        117,
        122
      ])

    data =
      Nx.tensor([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ])

    indptr_arr = from_tensor(indptr)
    indices_arr = from_tensor(indices)
    data_arr = from_tensor(data)

    assert EXGBoost.NIF.dmatrix_create_from_sparse(
             indptr_arr.binary,
             indptr_arr.typestr,
             Tuple.to_list(indptr_arr.shape),
             indptr_arr.readonly,
             indices_arr.binary,
             indices_arr.typestr,
             Tuple.to_list(indices_arr.shape),
             indices_arr.readonly,
             data_arr.binary,
             data_arr.typestr,
             Tuple.to_list(data_arr.shape),
             data_arr.readonly,
             ncols,
             config(),
             "csr"
           )
           |> unwrap!() !=
             :error

    assert EXGBoost.NIF.dmatrix_create_from_sparse(
             indptr_arr.binary,
             indptr_arr.typestr,
             Tuple.to_list(indptr_arr.shape),
             indptr_arr.readonly,
             indices_arr.binary,
             indices_arr.typestr,
             Tuple.to_list(indices_arr.shape),
             indices_arr.readonly,
             data_arr.binary,
             data_arr.typestr,
             Tuple.to_list(data_arr.shape),
             data_arr.readonly,
             ncols,
             config(),
             "csc"
           )
           |> unwrap!() !=
             :error

    {status, _} =
      EXGBoost.NIF.dmatrix_create_from_sparse(
        indptr_arr.binary,
        indptr_arr.typestr,
        Tuple.to_list(indptr_arr.shape),
        indptr_arr.readonly,
        indices_arr.binary,
        indices_arr.typestr,
        Tuple.to_list(indices_arr.shape),
        indices_arr.readonly,
        data_arr.binary,
        data_arr.typestr,
        Tuple.to_list(data_arr.shape),
        data_arr.readonly,
        ncols,
        config(),
        "csa"
      )

    assert status == :error
  end

  test "dmatrix_create_from_dense" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    arr = from_tensor(mat)
    shape = Tuple.to_list(arr.shape)

    assert EXGBoost.NIF.dmatrix_create_from_dense(
             arr.binary,
             arr.typestr,
             shape,
             arr.readonly,
             config()
           )
           |> unwrap!() != :error
  end

  test "dmatrix_set_str_feature_info" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    arr = from_tensor(mat)
    shape = Tuple.to_list(arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        arr.binary,
        arr.typestr,
        shape,
        arr.readonly,
        config()
      )
      |> unwrap!()

    assert EXGBoost.NIF.dmatrix_set_str_feature_info(dmat, ~c"feature_name", [
             ~c"name",
             ~c"color",
             ~c"length"
           ]) == :ok
  end

  test "dmatrix_get_str_feature_info" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    EXGBoost.NIF.dmatrix_set_str_feature_info(dmat, ~c"feature_name", [
      ~c"name",
      ~c"color",
      ~c"length"
    ])

    assert EXGBoost.NIF.dmatrix_get_str_feature_info(dmat, ~c"feature_name") |> unwrap!()
  end

  test "dmatrix_num_row" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    assert EXGBoost.NIF.dmatrix_num_row(dmat) |> unwrap! == 2
  end

  test "dmatrix_num_col" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    assert EXGBoost.NIF.dmatrix_num_col(dmat) |> unwrap! == 3
  end

  test "dmatrix_num_non_missing" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    assert EXGBoost.NIF.dmatrix_num_non_missing(dmat) |> unwrap! == 6
  end

  test "dmatrix_set_info_from_interface" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    labels = Nx.tensor([1.0, 0.0])
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    label_arr = from_tensor(labels)
    label_shape = Tuple.to_list(label_arr.shape)

    assert EXGBoost.NIF.dmatrix_set_info_from_interface(
             dmat,
             ~c"label",
             label_arr.binary,
             label_arr.typestr,
             label_shape,
             label_arr.readonly
           ) ==
             :ok

    assert EXGBoost.NIF.dmatrix_set_info_from_interface(
             dmat,
             ~c"unsupported",
             label_arr.binary,
             label_arr.typestr,
             label_shape,
             label_arr.readonly
           ) != :ok
  end

  test "dmatrix_save_binary" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    labels = Nx.tensor([1.0, 0.0])
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    label_arr = from_tensor(labels)
    label_shape = Tuple.to_list(label_arr.shape)

    EXGBoost.NIF.dmatrix_set_info_from_interface(
      dmat,
      ~c"label",
      label_arr.binary,
      label_arr.typestr,
      label_shape,
      label_arr.readonly
    )

    path = Path.join(System.tmp_dir!(), "test.buffer") |> String.to_charlist()
    assert EXGBoost.NIF.dmatrix_save_binary(dmat, path, 1) == :ok
  end

  test "dmatrix_get_float_info" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    weights = Nx.tensor([1.0, 0.0])
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    weights_arr = from_tensor(weights)
    weights_shape = Tuple.to_list(weights_arr.shape)

    EXGBoost.NIF.dmatrix_set_info_from_interface(
      dmat,
      ~c"feature_weights",
      weights_arr.binary,
      weights_arr.typestr,
      weights_shape,
      weights_arr.readonly
    )

    assert EXGBoost.NIF.dmatrix_get_float_info(dmat, ~c"feature_weights") |> unwrap!() ==
             Nx.to_list(weights)
  end

  test "dmatrix_get_data_as_csr" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    assert EXGBoost.NIF.dmatrix_get_data_as_csr(dmat, Jason.encode!(%{})) |> unwrap!() != :error
  end

  test "dmatrix_slice" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    # We do this because the C API uses non fixed-width types so we need to know the size they're expecting from int
    c_int_size = EXGBoost.NIF.get_int_size() |> unwrap!()
    tensor_size = c_int_size * 8

    dmatrix =
      EXGBoost.NIF.dmatrix_slice(
        dmat,
        Nx.to_binary(Nx.tensor([0, 1], type: {:s, tensor_size})),
        1
      )
      |> unwrap!()

    assert EXGBoost.NIF.dmatrix_num_row(dmatrix) |> unwrap!() == 2

    {status, _e} =
      EXGBoost.NIF.dmatrix_slice(
        dmat,
        Nx.to_binary(Nx.tensor([0, 1], type: {:s, tensor_size})),
        2
      )

    assert status == :error

    {status, _e} = EXGBoost.NIF.dmatrix_slice(dmat, Nx.to_binary(Nx.tensor([1.5, 1.6])), 2)

    assert status == :error
  end

  test "booster_create" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat2 = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat2_arr = from_tensor(mat2)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    mat2_shape = Tuple.to_list(mat2_arr.shape)

    dmat2 =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat2_arr.binary,
        mat2_arr.typestr,
        mat2_shape,
        mat2_arr.readonly,
        config()
      )
      |> unwrap!()

    assert EXGBoost.NIF.booster_create([dmat]) |> unwrap!() != :error
    assert EXGBoost.NIF.booster_create([]) |> unwrap!() != :error
    assert EXGBoost.NIF.booster_create([dmat, dmat2]) |> unwrap!() != :error
  end

  test "booster_get_num_feature" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    assert EXGBoost.NIF.booster_get_num_feature(booster) |> unwrap!() == 3
  end

  test "booster_set_str_feature_info" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()

    assert EXGBoost.NIF.booster_set_str_feature_info(booster, ~c"feature_name", [
             ~c"name",
             ~c"color",
             ~c"length"
           ]) == :ok
  end

  test "booster_get_str_feature_info" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()

    EXGBoost.NIF.booster_set_str_feature_info(booster, ~c"feature_name", [
      ~c"name",
      ~c"color",
      ~c"length"
    ])

    assert EXGBoost.NIF.booster_get_str_feature_info(booster, ~c"feature_name") |> unwrap!()
  end

  test "booster_feature_score" do
    # TODO: Make more robust test. This will just return an empty list
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    config = Jason.encode!(%{"importance_type" => "weight"})
    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()

    assert EXGBoost.NIF.booster_feature_score(booster, config) |> unwrap!() != :error
  end

  test "save model" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    json_file = Path.join(System.tmp_dir!(), "model.json") |> String.to_charlist()
    ubj_file = Path.join(System.tmp_dir!(), "model.ubj") |> String.to_charlist()
    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    assert EXGBoost.NIF.booster_save_model(booster, json_file) |> unwrap!() == :ok
    assert EXGBoost.NIF.booster_save_model(booster, ubj_file) |> unwrap!() == :ok
    assert File.exists?(json_file) and File.regular?(json_file)
    assert File.exists?(ubj_file) and File.regular?(ubj_file)
    assert File.rm(json_file) == :ok
    assert File.rm(ubj_file) == :ok
  end

  test "load model" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    json_file = Path.join(System.tmp_dir!(), "model.json") |> String.to_charlist()
    ubj_file = Path.join(System.tmp_dir!(), "model.ubj") |> String.to_charlist()
    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    assert EXGBoost.NIF.booster_save_model(booster, json_file) |> unwrap!() == :ok
    assert EXGBoost.NIF.booster_save_model(booster, ubj_file) |> unwrap!() == :ok
    assert File.exists?(json_file) and File.regular?(json_file)
    assert File.exists?(ubj_file) and File.regular?(ubj_file)
    assert EXGBoost.NIF.booster_load_model(json_file) |> unwrap!() != :error
    assert EXGBoost.NIF.booster_load_model(ubj_file) |> unwrap!() != :error
  end

  test "booster serialize" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    assert EXGBoost.NIF.booster_serialize_to_buffer(booster) |> unwrap!() != :error
  end

  test "booster deserialize" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    buffer = EXGBoost.NIF.booster_serialize_to_buffer(booster) |> unwrap!()
    EXGBoost.NIF.booster_deserialize_from_buffer(buffer)
    assert EXGBoost.NIF.booster_deserialize_from_buffer(buffer) |> unwrap!() != :error
  end

  test "save booster config" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    assert EXGBoost.NIF.booster_save_json_config(booster) |> unwrap!() != :error
  end

  test "load booster config" do
    mat = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mat_arr = from_tensor(mat)
    mat_shape = Tuple.to_list(mat_arr.shape)

    dmat =
      EXGBoost.NIF.dmatrix_create_from_dense(
        mat_arr.binary,
        mat_arr.typestr,
        mat_shape,
        mat_arr.readonly,
        config()
      )
      |> unwrap!()

    booster = EXGBoost.NIF.booster_create([dmat]) |> unwrap!()
    buf = EXGBoost.NIF.booster_save_json_config(booster) |> unwrap!()
    assert EXGBoost.NIF.booster_load_json_config(booster, buf) |> unwrap!() != :error
  end

  describe "Array Interface safety validations" do
    test "rejects invalid typestr format" do
      binary = <<1, 2, 3, 4, 5, 6, 7, 8>>

      result =
        EXGBoost.NIF.dmatrix_create_from_dense(
          binary,
          # Invalid format - should be like "<f4", "<i8", etc.
          "invalid_typestr",
          [2],
          true,
          config()
        )

      assert {:error, error_msg} = result
      assert to_string(error_msg) =~ "typestr"
    end

    test "accepts valid large unsigned shape dimension" do
      binary =
        <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native, 4.0::float-32-native>>

      result =
        EXGBoost.NIF.dmatrix_create_from_dense(
          binary,
          "<f4",
          # Valid: 4 elements * 4 bytes = 16 bytes
          [4],
          true,
          config()
        )

      assert {:ok, _dmatrix_ref} = result
    end

    test "rejects binary too small for specified shape" do
      # Only 8 bytes
      small_binary = <<1.0::float-32-native, 2.0::float-32-native>>

      result =
        EXGBoost.NIF.dmatrix_create_from_dense(
          small_binary,
          "<f4",
          # Would require 100*100*4 = 40,000 bytes!
          [100, 100],
          true,
          config()
        )

      assert {:error, error_msg} = result
      error_str = to_string(error_msg)
      assert error_str =~ "too small" or error_str =~ "size"
    end

    test "rejects invalid readonly value" do
      binary = <<1.0::float-32-native, 2.0::float-32-native>>

      result =
        EXGBoost.NIF.dmatrix_create_from_dense(
          binary,
          "<f4",
          [2],
          # Invalid: string instead of boolean atom
          "not_a_boolean",
          config()
        )

      assert {:error, error_msg} = result
      error_str = to_string(error_msg)
      assert error_str =~ "boolean" or error_str =~ "readonly"
    end

    test "succeeds with valid parameters" do
      binary =
        <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native, 4.0::float-32-native>>

      result =
        EXGBoost.NIF.dmatrix_create_from_dense(
          binary,
          "<f4",
          # 2*2*4 = 16 bytes (matches binary size)
          [2, 2],
          true,
          config()
        )

      assert {:ok, _dmatrix_ref} = result
    end

    test "validates typestr with different endianness markers" do
      binary = <<1.0::float-32-native, 2.0::float-32-native>>

      # Little-endian should work
      assert {:ok, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "<f4",
                 [2],
                 true,
                 config()
               )

      # Non-endian should work
      assert {:ok, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "|i4",
                 [2],
                 true,
                 config()
               )

      # Invalid endianness marker should fail
      assert {:error, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "?f4",
                 [2],
                 true,
                 config()
               )
    end

    test "validates shape with overflow protection" do
      binary = <<1.0::float-32-native, 2.0::float-32-native>>

      # Extremely large shape that would overflow
      # Note: This might not fail if XGBoost itself fails first,
      # but our validation should catch reasonable overflows
      result =
        EXGBoost.NIF.dmatrix_create_from_dense(
          binary,
          "<f4",
          # Would require 4TB!
          [1_000_000, 1_000_000],
          true,
          config()
        )

      assert {:error, _error_msg} = result
    end

    test "strict boolean validation requires atoms" do
      binary = <<1.0::float-32-native, 2.0::float-32-native>>

      # Integer should fail
      assert {:error, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "<f4",
                 [2],
                 1,
                 config()
               )

      # String should fail
      assert {:error, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "<f4",
                 [2],
                 "true",
                 config()
               )

      # Only true/false atoms should work
      assert {:ok, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "<f4",
                 [2],
                 true,
                 config()
               )

      assert {:ok, _} =
               EXGBoost.NIF.dmatrix_create_from_dense(
                 binary,
                 "<f4",
                 [2],
                 false,
                 config()
               )
    end
  end

  # Helper to build config with missing value
  defp config(opts \\ []) do
    Map.merge(%{"missing" => -1.0}, Enum.into(opts, %{}))
    |> Jason.encode!()
  end
end
