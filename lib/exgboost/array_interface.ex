defmodule EXGBoost.ArrayInterface do
  @moduledoc false

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
          readonly: boolean(),
          binary: binary() | nil
        }

  @enforce_keys [:typestr, :shape, :readonly]
  defstruct [
    :typestr,
    :shape,
    :readonly,
    :binary,
    version: 3
  ]

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(
          %{
            typestr: typestr,
            shape: shape,
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
            readonly: readonly,
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
      {"data", [_address, readonly]}, acc ->
        [{:readonly, readonly} | acc]

      {:data, [_address, readonly]}, acc ->
        [{:readonly, readonly} | acc]

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
        %{readonly: true, shape: [2, 3], typestr: "<i8", version: 3}
        >
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

    %__MODULE__{
      typestr: type_char,
      shape: Nx.shape(tensor),
      readonly: true,
      binary: binary
    }
  end

  @spec get_tensor(EXGBoost.ArrayInterface.t()) :: Nx.Tensor.t()
  def get_tensor(%__MODULE__{binary: binary, typestr: typestr, shape: shape}) when is_binary(binary) do
    # Parse typestr to get Nx type
    <<_endian::utf8, char_code::binary-size(1), bytes::binary>> = typestr
    bit_width = String.to_integer(bytes) * 8

    nx_type =
      case char_code do
        "i" -> {:s, bit_width}
        "u" -> {:u, bit_width}
        "f" -> {:f, bit_width}
        "c" -> {:c, bit_width}
      end

    Nx.from_binary(binary, nx_type) |> Nx.reshape(shape)
  end

  def get_tensor(%__MODULE__{binary: nil}) do
    raise ArgumentError, """
    Cannot reconstruct tensor from ArrayInterface without binary data.

    ArrayInterface instances must include the binary data field.
    If you're seeing this error, ensure the ArrayInterface was created with
    from_tensor/1 or includes binary data from a NIF (like get_quantile_cut).
    """
  end
end
