defmodule EXGBoost.ArrayInterface do
  @moduledoc false
  alias EXGBoost.Internal

  @typedoc """
  The XGBoost C API uses and is moving towards mainly supporting the use of
  JSON-Encoded NumPy ArrayInterface format to pass data to and from the C API. This struct
  is used to represent the ArrayInterface format.

  If you wish to use the EXGBoost.NIF library directly, this will be the desired format
  to pass Nx.Tensors to the NIFs. Use of the EXGBoost.NIF library directly is not recommended
  unless you are familiar with the XGBoost C API and the EXGBoost.NIF library.

  See https://numpy.org/doc/stable/reference/arrays.interface.html for more information on
  the ArrayInterface protocol.
  """
  @type t :: %__MODULE__{
          typestr: String.t(),
          shape: tuple(),
          address: non_neg_integer(),
          readonly: boolean(),
          tensor: Nx.Tensor.t() | nil,
          binary: binary() | nil
        }

  @enforce_keys [:typestr, :shape, :address, :readonly]
  defstruct [
    :typestr,
    :shape,
    :address,
    :readonly,
    :tensor,
    :binary,
    version: 3
  ]

  defimpl Jason.Encoder do
    def encode(
          %{
            typestr: typestr,
            shape: shape,
            address: address,
            readonly: readonly,
            version: version
          },
          opts
        ) do
      Jason.Encode.map(
        %{
          typestr: typestr,
          shape: Tuple.to_list(shape),
          data: [address, readonly],
          version: version
        },
        opts
      )
    end
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(
          %{
            typestr: typestr,
            shape: shape,
            address: address,
            readonly: readonly,
            version: version
          },
          opts
        ) do
      concat([
        "#ArrayInterface<",
        line(),
        to_doc(
          %{
            typestr: typestr,
            shape: Tuple.to_list(shape),
            data: [address, readonly],
            version: version
          },
          opts
        ),
        line(),
        ">"
      ])
    end
  end

  def from_map(%{} = interface) do
    interface
    |> Enum.reduce([], fn
      {"data", [address, readonly]}, acc ->
        [{:address, address} | [{:readonly, readonly} | acc]]

      {:data, [address, readonly]}, acc ->
        [{:address, address} | [{:readonly, readonly} | acc]]

      {"shape", shape}, acc ->
        [{:shape, List.to_tuple(shape)} | acc]

      {:shape, shape}, acc ->
        [{:shape, List.to_tuple(shape)} | acc]

      {"typestr", typestr}, acc ->
        [{:typestr, typestr} | acc]

      {:typestr, typestr}, acc ->
        [{:typestr, typestr} | acc]

      {"version", version}, acc ->
        [{:version, version} | acc]

      {:version, version}, acc ->
        [{:version, version} | acc]

      {"tensor", tensor}, acc ->
        [{:tensor, tensor} | acc]

      {:tensor, tensor}, acc ->
        [{:tensor, tensor} | acc]

      {"binary", binary}, acc ->
        [{:binary, binary} | acc]

      {:binary, binary}, acc ->
        [{:binary, binary} | acc]

      {_key, _value}, acc ->
        acc
    end)
    |> then(&struct(__MODULE__, &1))
  end

  @doc """
  This function is used to convert Nx.Tensors to the ArrayInterface format.

  Example:
    iex> EXGBoost.ArrayInterface.from_tensor(Nx.tensor([[1,2,3],[4,5,6]]))
        #ArrayInterface<
        %{data: [4418559984, true], shape: [2, 3], typestr: "<i8", version: 3}
  """
  @spec from_tensor(Nx.Tensor.t()) :: %__MODULE__{}
  def from_tensor(%Nx.Tensor{type: t_type} = tensor) do
    type_char =
      case t_type do
        {:s, width} ->
          "<i#{div(width, 8)}"

        # TODO: Use V typestr to handle other data types
        {:bf, _width} ->
          raise ArgumentError,
                "Invalid tensor type -- #{inspect(t_type)} not supported by EXGBoost"

        {tensor_type, type_width} ->
          "<#{Atom.to_string(tensor_type)}#{div(type_width, 8)}"
      end

    binary = Nx.to_binary(tensor)

    tensor_addr =
      EXGBoost.NIF.get_binary_address(binary) |> EXGBoost.Internal.unwrap!()

    %__MODULE__{
      typestr: type_char,
      shape: Nx.shape(tensor),
      address: tensor_addr,
      readonly: true,
      tensor: tensor,
      binary: binary
    }
  end

  @spec get_tensor(EXGBoost.ArrayInterface.t()) :: Nx.Tensor.t()
  def get_tensor(%__MODULE__{tensor: nil} = arr_int) do
    num_items = arr_int.shape |> Tuple.to_list() |> Enum.product()
    <<endianess::utf8, char_code::binary-size(1), bytes::binary>> = arr_int.typestr

    if endianess not in [?<, ?|] do
      raise ArgumentError,
            "Unsupported endianness in typestr #{inspect(arr_int.typestr)}. " <>
              "Expected little-endian ('<') or non-endian ('|')."
    end

    bit_width = String.to_integer(bytes) * 8

    nx_type =
      case char_code do
        "i" ->
          {:s, bit_width}

        "u" ->
          {:u, bit_width}

        "f" ->
          {:f, bit_width}

        "c" ->
          {:c, bit_width}

        other ->
          raise ArgumentError,
                "Unsupported typestr code #{inspect(other)} in #{inspect(arr_int.typestr)}"
      end

    tensor_bin =
      EXGBoost.NIF.get_binary_from_address(arr_int.address, String.to_integer(bytes) * num_items)
      |> Internal.unwrap!()

    Nx.from_binary(
      tensor_bin,
      nx_type
    )
    |> Nx.reshape(arr_int.shape)
  end

  def get_tensor(%__MODULE__{tensor: %Nx.Tensor{} = tensor}) do
    tensor
  end
end
