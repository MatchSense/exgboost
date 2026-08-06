defmodule CovtypeBenchmark do
  @url "https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz"
  @data_dir Path.expand("data/covtype", __DIR__)
  @gzip_path Path.join(@data_dir, "covtype.data.gz")
  @x_path Path.join(@data_dir, "x.float32.bin")
  @y_path Path.join(@data_dir, "y.uint32.bin")

  @rows 581_012
  @features 54
  @batch_size 50_000

  def run do
    File.mkdir_p!(@data_dir)

    download!()
    convert_unless_present!()

    x =
      @x_path
      |> File.read!()
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({@rows, @features})

    # Labels in the source are 1..7. XGBoost expects 0..6.
    y =
      @y_path
      |> File.read!()
      |> Nx.from_binary({:u, 32})
      |> Nx.reshape({@rows})

    IO.puts("Loaded x=#{inspect(Nx.shape(x))}, y=#{inspect(Nx.shape(y))}")

    rounds = env_integer("ROUNDS", 500)
    max_depth = env_integer("MAX_DEPTH", 8)

    devices =
      case System.get_env("DEVICE", "both") do
        "cpu" -> [:cpu]
        "cuda" -> [:cuda]
        "both" -> [:cpu, :cuda]
        other -> raise "DEVICE must be cpu, cuda, or both; got #{inspect(other)}"
      end

    results =
      for device <- devices, into: %{} do
        seconds = train(x, y, device, rounds, max_depth)
        {device, seconds}
      end

    case results do
      %{cpu: cpu, cuda: cuda} ->
        IO.puts("\nCUDA speed-up: #{Float.round(cpu / cuda, 2)}x")

      _ ->
        :ok
    end
  end

  defp download! do
    if File.exists?(@gzip_path) do
      IO.puts("Using #{@gzip_path}")
    else
      IO.puts("Downloading Covertype...")

      {_, status} =
        System.cmd(
          "curl",
          ["-fL", "--retry", "5", "--continue-at", "-", "-o", @gzip_path, @url],
          into: IO.stream()
        )

      if status != 0 do
        raise "Download failed with status #{status}"
      end
    end
  end

  defp convert_unless_present! do
    expected_x_bytes = @rows * @features * 4
    expected_y_bytes = @rows * 4

    if valid_file?(@x_path, expected_x_bytes) and
         valid_file?(@y_path, expected_y_bytes) do
      IO.puts("Using existing converted binaries")
    else
      convert!()
    end
  end

  defp convert! do
    IO.puts("Converting compressed CSV to binary...")

    {:ok, x_file} = File.open(@x_path, [:write, :binary])
    {:ok, y_file} = File.open(@y_path, [:write, :binary])

    try do
      rows_written =
        @gzip_path
        |> File.stream!(:line, [:compressed])
        |> Stream.with_index(1)
        |> Stream.chunk_every(@batch_size)
        |> Enum.reduce(0, fn batch, total ->
          {x_iodata, y_iodata} = encode_batch!(batch)

          :ok = IO.binwrite(x_file, x_iodata)
          :ok = IO.binwrite(y_file, y_iodata)

          next_total = total + length(batch)
          IO.write("\rConverted #{format_integer(next_total)} rows")
          next_total
        end)

      IO.puts("")

      if rows_written != @rows do
        raise "Expected #{@rows} rows, converted #{rows_written}"
      end
    after
      File.close(x_file)
      File.close(y_file)
    end

    IO.puts("Conversion complete")
  end

  defp encode_batch!(batch) do
    {x_rows, y_rows} =
      Enum.reduce(batch, {[], []}, fn {line, line_number}, {x_acc, y_acc} ->
        columns =
          line
          |> String.trim_trailing()
          |> String.split(",")

        if length(columns) != @features + 1 do
          raise "Line #{line_number}: expected 55 columns, got #{length(columns)}"
        end

        {feature_strings, [label_string]} = Enum.split(columns, @features)

        features =
          Enum.map(feature_strings, &parse_number!(&1, line_number))

        label =
          case Integer.parse(label_string) do
            {value, ""} when value in 1..7 -> value - 1
            _ -> raise "Line #{line_number}: invalid label #{inspect(label_string)}"
          end

        x_row =
          for value <- features do
            <<value::float-32-little>>
          end

        {
          [x_row | x_acc],
          [<<label::unsigned-32-little>> | y_acc]
        }
      end)

    {
      Enum.reverse(x_rows),
      Enum.reverse(y_rows)
    }
  end

  defp train(x, y, device, rounds, max_depth) do
    IO.puts("\nTraining on #{device}...")

    # Warm the NIF and initialise CUDA outside the timed section.
    warm_rows = 10_000

    _warmup =
      EXGBoost.train(
        x[[0..(warm_rows - 1), ..]],
        y[[0..(warm_rows - 1)]],
        device: device,
        tree_method: :hist,
        objective: :multi_softprob,
        num_class: 7,
        num_boost_rounds: 2,
        verbose_eval: false
      )

    {microseconds, _booster} =
      :timer.tc(fn ->
        EXGBoost.train(
          x,
          y,
          device: device,
          tree_method: :hist,
          objective: :multi_softprob,
          eval_metric: :mlogloss,
          num_class: 7,
          num_boost_rounds: rounds,
          max_depth: max_depth,
          max_bin: 256,
          seed: 42,
          verbose_eval: false
        )
      end)

    seconds = microseconds / 1_000_000
    IO.puts("#{device}: #{Float.round(seconds, 3)} seconds")
    seconds
  end

  defp valid_file?(path, expected_size) do
    case File.stat(path) do
      {:ok, %{size: ^expected_size}} -> true
      _ -> false
    end
  end

  defp env_integer(name, default) do
    case System.get_env(name) do
      nil -> default
      value -> String.to_integer(value)
    end
  end

  defp format_integer(number) do
    number
    |> Integer.to_string()
    |> String.reverse()
    |> String.replace(~r/(\d{3})(?=\d)/, "\\1,")
    |> String.reverse()
  end

  defp parse_number!(value, line_number) do
    case Integer.parse(value) do
      {number, ""} ->
        number * 1.0

      _ ->
        case Float.parse(value) do
          {number, ""} ->
            number

          _ ->
            raise "Line #{line_number}: invalid value #{inspect(value)}"
        end
    end
end
end

CovtypeBenchmark.run()
