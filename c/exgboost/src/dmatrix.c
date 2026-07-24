#include "dmatrix.h"
#include <inttypes.h>
#include <string.h>

static ERL_NIF_TERM make_DMatrix_resource(ErlNifEnv *env,
                                          DMatrixHandle handle) {
  ERL_NIF_TERM ret = -1;
  DMatrixHandle **resource =
      enif_alloc_resource(DMatrix_RESOURCE_TYPE, sizeof(DMatrixHandle *));
  if (resource != NULL) {
    *resource = handle;
    // BEAM resource now owns the handle and releases it in resource cleanup.
    ret = exg_ok(env, enif_make_resource(env, resource));
    enif_release_resource(resource);
  } else {
    ret = exg_error(env, "Failed to allocate memory for XGBoost DMatrix");
  }
  return ret;
}

// Deprecated since XGBoost 2.0.0
ERL_NIF_TERM EXGDMatrixCreateFromFile(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  char *fname = NULL;
  char *format = NULL;
  int silent = 0;
  int result = -1;
  DMatrixHandle handle;
  ERL_NIF_TERM ret = 0;
  if (argc != 2) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!exg_get_string(env, argv[0], &fname)) {
    ret = exg_error(env, "File name must be a string");
    goto END;
  }
  if (!enif_get_int(env, argv[1], &silent)) {
    ret = exg_error(env, "Silent must be an integer");
    goto END;
  }
  result = XGDMatrixCreateFromFile(fname, 1, &handle);
  if (result == 0) {
    ret = make_DMatrix_resource(env, handle);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (fname != NULL) {
    enif_free(fname);
    fname = NULL;
  }
  if (format != NULL) {
    enif_free(format);
    format = NULL;
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixCreateFromURI(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  char *config = NULL;
  DMatrixHandle handle;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 1) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!exg_get_string(env, argv[0], &config)) {
    ret = exg_error(env, "Config must be a string");
    goto END;
  }
  result = XGDMatrixCreateFromURI(config, &handle);
  if (result == 0) {
    ret = make_DMatrix_resource(env, handle);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (config != NULL) {
    enif_free(config);
    config = NULL;
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixCreateFromMat(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  ErlNifBinary bin;
  int result = -1;
  float *mat = NULL;
  int nrow = 0;
  int ncol = 0;
  int num_floats = 0;
  double missing = 0.0;
  DMatrixHandle handle;
  ERL_NIF_TERM ret = 0;
  if (argc != 4) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_inspect_binary(env, argv[0], &bin)) {
    ret = exg_error(env, "Data must be a binary");
    goto END;
  }
  if (!enif_get_int(env, argv[1], &nrow)) {
    ret = exg_error(env, "Nrow must be an integer");
    goto END;
  }
  if (!enif_get_int(env, argv[2], &ncol)) {
    ret = exg_error(env, "Ncol must be an integer");
    goto END;
  }
  if (!enif_get_double(env, argv[3], &missing)) {
    ret = exg_error(env, "Missing must be a float");
    goto END;
  }
  mat = (float *)bin.data;
  num_floats = bin.size / sizeof(float);
  if (num_floats != nrow * ncol) {
    ret = exg_error(env, "Data size does not match nrow and ncol");
    goto END;
  }
  // The DMatrix wlil keep ahold of this data, so we don't need to free it
  // Will be freed when DMatrix is freed in resource destructor
  result = XGDMatrixCreateFromMat(mat, (bst_ulong)nrow, (bst_ulong)ncol,
                                  missing, &handle);
  if (result == 0) {
    ret = make_DMatrix_resource(env, handle);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  return ret;
}

ERL_NIF_TERM EXGDMatrixCreateFromSparse(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  int result = -1;
  char *indptr_interface = NULL;
  char *indices_interface = NULL;
  char *data_interface = NULL;
  const char *error_msg = NULL;
  int n = 0;
  char *config = NULL;
  char *format = NULL;
  DMatrixHandle handle;
  ERL_NIF_TERM ret = 0;

  if (argc != 15) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }

  // Extract ArrayInterfaces with fresh addresses: (binary, typestr, shape, readonly) per array
  if (!exg_build_array_interface_json(env, argv[0], argv[1], argv[2], argv[3], &indptr_interface, &error_msg)) {
    ret = exg_error(env, error_msg ? error_msg : "Failed to extract indptr ArrayInterface");
    goto END;
  }

  if (!exg_build_array_interface_json(env, argv[4], argv[5], argv[6], argv[7], &indices_interface, &error_msg)) {
    ret = exg_error(env, error_msg ? error_msg : "Failed to extract indices ArrayInterface");
    goto END;
  }

  if (!exg_build_array_interface_json(env, argv[8], argv[9], argv[10], argv[11], &data_interface, &error_msg)) {
    ret = exg_error(env, error_msg ? error_msg : "Failed to extract data ArrayInterface");
    goto END;
  }

  if (!enif_get_int(env, argv[12], &n)) {
    ret = exg_error(env, "Ncol must be an integer");
    goto END;
  }

  if (!exg_get_string(env, argv[13], &config)) {
    ret = exg_error(env, "Config must be a string");
    goto END;
  }

  if (!exg_get_string(env, argv[14], &format)) {
    ret = exg_error(env, "Format must be a string");
    goto END;
  }
  if (strcmp(format, "csr") == 0) {
    result = XGDMatrixCreateFromCSR(indptr_interface, indices_interface,
                                    data_interface, n, config, &handle);
  } else if (strcmp(format, "csc") == 0) {
    result = XGDMatrixCreateFromCSC(indptr_interface, indices_interface,
                                    data_interface, n, config, &handle);
  } else {
    ret = exg_error(env, "Format must in ['csr','csc']");
    goto END;
  }
  if (result == 0) {
    ret = make_DMatrix_resource(env, handle);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (config != NULL) {
    enif_free(config);
    config = NULL;
  }
  if (indptr_interface != NULL) {
    enif_free(indptr_interface);
    indptr_interface = NULL;
  }
  if (indices_interface != NULL) {
    enif_free(indices_interface);
    indices_interface = NULL;
  }
  if (data_interface != NULL) {
    enif_free(data_interface);
    data_interface = NULL;
  }
  if (format != NULL) {
    enif_free(format);
    format = NULL;
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixCreateFromDense(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  int result = -1;
  char *array_interface = NULL;
  const char *error_msg = NULL;
  char *config = NULL;
  DMatrixHandle out;
  ERL_NIF_TERM ret = 0;

  if (argc != 5) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }

  // Build ArrayInterface JSON from components: (binary, typestr, shape, readonly)
  if (!exg_build_array_interface_json(env, argv[0], argv[1], argv[2], argv[3], &array_interface, &error_msg)) {
    ret = exg_error(env, error_msg ? error_msg : "Failed to extract ArrayInterface");
    goto END;
  }

  if (!exg_get_string(env, argv[4], &config)) {
    ret = exg_error(env, "Config must be a JSON-Encoded string");
    goto END;
  }
  result = XGDMatrixCreateFromDense(array_interface, config, &out);
  if (0 == result) {
    ret = make_DMatrix_resource(env, out);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (array_interface != NULL) {
    enif_free(array_interface);
  }
  if (config != NULL) {
    enif_free(config);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixSetStrFeatureInfo(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char **features = NULL;
  unsigned num_features = 0;
  char *field = NULL;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 3) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &field)) {
    ret = exg_error(env, "Field must be a string");
    goto END;
  }
  if (!exg_get_string_list(env, argv[2], &features, &num_features)) {
    ret = exg_error(env, "Features must be a list");
    goto END;
  }
  if (strcmp(field, "feature_type") != 0 &&
      strcmp(field, "feature_name") != 0) {
    ret = exg_error(env, "Field must be in ['feature_type', 'feature_name']");
    goto END;
  }
  handle = *resource;
  // XGBoost reads features during the call; caller keeps ownership.
  result = XGDMatrixSetStrFeatureInfo(handle, field,
                                      (const char **)features, num_features);
  if (result == 0) {
    ret = ok_atom(env);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  // Helper-allocated buffers must be reclaimed on all paths.
  if (features != NULL) {
    exg_free_string_list(features, num_features);
  }
  if (field != NULL) {
    enif_free(field);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixGetStrFeatureInfo(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char const **c_out_features = NULL;
  bst_ulong out_size = 0;
  char *field = NULL;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 2) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &field)) {
    ret = exg_error(env, "Field must be a string");
    goto END;
  }
  if (strcmp(field, "feature_type") != 0 &&
      strcmp(field, "feature_name") != 0) {
    ret = exg_error(env, "Field must be in ['feature_type', 'feature_name']");
    goto END;
  }
  handle = *resource;
  result =
      XGDMatrixGetStrFeatureInfo(handle, field, &out_size, &c_out_features);
  if (result == 0) {
    // Check VLA size fits in size_t
    if (out_size > SIZE_MAX / sizeof(ERL_NIF_TERM)) {
      ret = exg_error(env, "Result is too large");
      goto END;
    }
    ERL_NIF_TERM arr[out_size];
    for (bst_ulong i = 0; i < out_size; ++i) {
      // enif_make_string materializes a BEAM term; no temporary C copy needed.
      arr[i] = enif_make_string(env, c_out_features[i], ERL_NIF_LATIN1);
    }
    ret = exg_ok(env, enif_make_list_from_array(env, arr, (size_t)out_size));
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (field != NULL) {
    enif_free(field);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixSetDenseInfo(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  ErlNifBinary data_bin;
  DMatrixHandle **resource = NULL;
  char *field = NULL;
  bst_ulong size = 0;
  int type = -1;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 5) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &field)) {
    ret = exg_error(env, "Field must be a string");
    goto END;
  }
  if (!enif_inspect_binary(env, argv[2], &data_bin)) {
    ret = exg_error(env, "Data must be a binary");
    goto END;
  }
  ErlNifUInt64 size_arg;
  if (!enif_get_uint64(env, argv[3], &size_arg)) {
    ret = exg_error(env, "Size must be a non-negative integer");
    goto END;
  }
  size = (bst_ulong)size_arg;
  if (!enif_get_int(env, argv[4], &type)) {
    ret = exg_error(env, "Type must be an integer");
    goto END;
  }
  if (strcmp(field, "label") != 0 && strcmp(field, "weight") != 0 &&
      strcmp(field, "base_margin") != 0 && strcmp(field, "group") != 0 &&
      strcmp(field, "label_lower_bound") != 0 &&
      strcmp(field, "label_upper_bound") != 0 &&
      strcmp(field, "feature_weights") != 0) {
    ret = exg_error(env, "Field must be in ['label', 'weight', "
                         "'base_margin','group','label_lower_bound','label_"
                         "upper_bound','feature_weights']");
    goto END;
  }
  if (type < 1 && type > 4) {
    ret = exg_error(env, "Type must be in [1..4]");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixSetDenseInfo(handle, field, data_bin.data, size, type);
  if (result == 0) {
    ret = ok_atom(env);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (field != NULL) {
    enif_free(field);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixNumRow(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  bst_ulong out = 0;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 1) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixNumRow(handle, &out);
  if (result == 0) {
    ret = exg_ok(env, enif_make_uint64(env, (ErlNifUInt64)out));
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  return ret;
}

ERL_NIF_TERM EXGDMatrixNumCol(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  bst_ulong out = 0;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 1) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixNumCol(handle, &out);
  if (result == 0) {
    ret = exg_ok(env, enif_make_uint64(env, (ErlNifUInt64)out));
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  return ret;
}

ERL_NIF_TERM EXGDMatrixNumNonMissing(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  bst_ulong out = 0;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 1) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixNumNonMissing(handle, &out);
  if (result == 0) {
    ret = exg_ok(env, enif_make_uint64(env, (ErlNifUInt64)out));
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  return ret;
}

ERL_NIF_TERM EXGDMatrixSetInfoFromInterface(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char *field = NULL;
  char *data_interface = NULL;
  const char *error_msg = NULL;
  int result = -1;
  ERL_NIF_TERM ret = 0;

  if (argc != 6) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }

  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }

  if (!exg_get_string(env, argv[1], &field)) {
    ret = exg_error(env, "Field must be a string");
    goto END;
  }

  // Build ArrayInterface JSON from components: (binary, typestr, shape, readonly)
  if (!exg_build_array_interface_json(env, argv[2], argv[3], argv[4], argv[5], &data_interface, &error_msg)) {
    ret = exg_error(env, error_msg ? error_msg : "Failed to extract data ArrayInterface");
    goto END;
  }

  if (strcmp(field, "label") != 0 && strcmp(field, "weight") != 0 &&
      strcmp(field, "base_margin") != 0 && strcmp(field, "group") != 0 &&
      strcmp(field, "label_lower_bound") != 0 &&
      strcmp(field, "label_upper_bound") != 0 &&
      strcmp(field, "feature_weights") != 0) {
    ret = exg_error(env, "Field must be in ['label', 'weight', "
                         "'base_margin','group','label_lower_bound','label_"
                         "upper_bound','feature_weights']");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixSetInfoFromInterface(handle, field, data_interface);
  if (result == 0) {
    ret = ok_atom(env);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (field != NULL) {
    enif_free(field);
  }
  if (data_interface != NULL) {
    enif_free(data_interface);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixSaveBinary(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char *fname = NULL;
  int silent = 0;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 3) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &fname)) {
    ret = exg_error(env, "File name must be a string");
    goto END;
  }
  if (!enif_get_int(env, argv[2], &silent)) {
    ret = exg_error(env, "Silent must be an integer");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixSaveBinary(handle, fname, silent);
  if (result == 0) {
    ret = ok_atom(env);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (fname != NULL) {
    enif_free(fname);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixGetFloatInfo(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char *field = NULL;
  const float *out = NULL;
  bst_ulong len = 0;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  ERL_NIF_TERM *arr = NULL;

  if (argc != 2) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &field)) {
    ret = exg_error(env, "Field must be a string");
    goto END;
  }
  if (strcmp(field, "label") != 0 && strcmp(field, "weight") != 0 &&
      strcmp(field, "base_margin") != 0 && strcmp(field, "label_lower_bound") &&
      strcmp(field, "label_upper_bound") &&
      strcmp(field, "feature_weights") != 0) {
    ret = exg_error(env, "Field must be in ['label', 'weight', "
                         "'base_margin','label_lower_bound','label_"
                         "upper_bound','feature_weights']");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixGetFloatInfo(handle, field, &len, &out);
  if (result == 0) {
    arr = enif_alloc(sizeof(ERL_NIF_TERM) * len);
    for (int i = 0; i < len; i++) {
      // Values are copied into BEAM-managed terms here.
      arr[i] = enif_make_double(env, out[i]);
    }
    ret = exg_ok(env, enif_make_list_from_array(env, arr, len));
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (field != NULL) {
    enif_free(field);
  }
  if (arr != NULL) {
    enif_free(arr);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixGetUIntInfo(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char *field = NULL;
  const unsigned *out = NULL;
  bst_ulong len = 0;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  ERL_NIF_TERM *arr = NULL;

  if (argc != 2) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &field)) {
    ret = exg_error(env, "Field must be a string");
    goto END;
  }
  if (strcmp(field, "group_ptr") != 0) {
    ret = exg_error(env, "Field must be in ['group_ptr']");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixGetUIntInfo(handle, field, &len, &out);
  if (result == 0) {
    arr = enif_alloc(sizeof(ERL_NIF_TERM) * len);
    for (int i = 0; i < len; i++) {
      // Values are copied into BEAM-managed terms here.
      arr[i] = enif_make_uint(env, out[i]);
    }
    ret = exg_ok(env, enif_make_list_from_array(env, arr, len));
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  if (field != NULL) {
    enif_free(field);
  }
  if (arr != NULL) {
    enif_free(arr);
  }
  return ret;
}

ERL_NIF_TERM EXGDMatrixGetDataAsCSR(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  bst_ulong num_non_missing = 0;
  bst_ulong num_rows = 0;
  char *config = NULL;
  bst_ulong *out_indptr = NULL;
  unsigned *out_indices = NULL;
  float *out_data = NULL;
  ERL_NIF_TERM *indptr = NULL;
  ERL_NIF_TERM *indices = NULL;
  ERL_NIF_TERM *data = NULL;
  int result = -1;
  ERL_NIF_TERM ret = 0;
  if (argc != 2) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &config)) {
    ret = exg_error(env, "Config must be a JSON-Encoded string");
    goto END;
  }
  handle = *resource;
  result = XGDMatrixNumRow(handle, &num_rows);
  if (result != 0) {
    ret = exg_error(env, XGBGetLastError());
    goto END;
  }
  result = XGDMatrixNumNonMissing(handle, &num_non_missing);
  if (result != 0) {
    ret = exg_error(env, XGBGetLastError());
    goto END;
  }
  // Check allocation sizes fit in size_t
  if (num_rows > SIZE_MAX / sizeof(bst_ulong) - 1 ||
      num_non_missing > SIZE_MAX / sizeof(unsigned) ||
      num_non_missing > SIZE_MAX / sizeof(float)) {
    ret = exg_error(env, "Matrix is too large");
    goto END;
  }
  out_indptr = malloc(sizeof(bst_ulong) * (size_t)(num_rows + 1));
  out_indices = malloc(sizeof(unsigned) * (size_t)num_non_missing);
  out_data = malloc(sizeof(float) * (size_t)num_non_missing);
  if (!out_indptr || !out_indices || !out_data) {
    ret = exg_error(env, "Failed to allocate memory");
    goto END;
  }
  result =
      XGDMatrixGetDataAsCSR(handle, config, out_indptr, out_indices, out_data);
  if (result != 0) {
    ret = exg_error(env, XGBGetLastError());
    goto END;
  }
  // Check enif_alloc sizes fit in size_t
  if (num_rows > SIZE_MAX / sizeof(ERL_NIF_TERM) - 1 ||
      num_non_missing > SIZE_MAX / sizeof(ERL_NIF_TERM)) {
    ret = exg_error(env, "Matrix is too large");
    goto END;
  }
  indptr = enif_alloc(sizeof(ERL_NIF_TERM) * (size_t)(num_rows + 1));
  indices = enif_alloc(sizeof(ERL_NIF_TERM) * (size_t)num_non_missing);
  data = enif_alloc(sizeof(ERL_NIF_TERM) * (size_t)num_non_missing);
  if (!indptr || !indices || !data) {
    ret = exg_error(env, "Failed to allocate memory");
    goto END;
  }
  for (bst_ulong i = 0; i < num_rows + 1; ++i) {
    indptr[i] = enif_make_uint64(env, (ErlNifUInt64)out_indptr[i]);
  }
  for (bst_ulong i = 0; i < num_non_missing; ++i) {
    indices[i] = enif_make_uint(env, out_indices[i]);
    data[i] = enif_make_double(env, out_data[i]);
  }
  ret =
      exg_ok(env, enif_make_tuple3(
                      env, enif_make_list_from_array(env, indptr, (size_t)(num_rows + 1)),
                      enif_make_list_from_array(env, indices, (size_t)num_non_missing),
                      enif_make_list_from_array(env, data, (size_t)num_non_missing)));
END:
  // Mixed allocators: enif_free for enif_alloc buffers, free for malloc buffers.
  if (config != NULL) {
    enif_free(config);
    config = NULL;
  }
  if (out_indptr != NULL) {
    free(out_indptr);
    out_indptr = NULL;
  }
  if (out_indices != NULL) {
    free(out_indices);
    out_indices = NULL;
  }
  if (out_data != NULL) {
    free(out_data);
    out_data = NULL;
  }
  if (indptr != NULL) {
    enif_free(indptr);
  }
  if (indices != NULL) {
    enif_free(indices);
  }
  if (data != NULL) {
    enif_free(data);
  }
  return ret;
};

ERL_NIF_TERM EXGDMatrixSliceDMatrix(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  ErlNifBinary bin;
  DMatrixHandle out;
  int result = -1;
  int allow_groups = 0;
  ERL_NIF_TERM ret = -1;
  if (argc != 3) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE,
                         (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!enif_inspect_binary(env, argv[1], &bin)) {
    ret = exg_error(env, "Indices must be a binary of ints");
    goto END;
  }
  if (!enif_get_int(env, argv[2], &allow_groups)) {
    ret = exg_error(env, "allow_groups must be an int");
    goto END;
  }
  if (bin.size % sizeof(int) != 0) {
    ret = exg_error(env, "Indices must be a binary of ints");
    goto END;
  }
  if (bin.size == 0) {
    ret = exg_error(env, "Indices must be a binary of ints (non-empty))");
    goto END;
  }
  if (allow_groups != 0 && allow_groups != 1) {
    ret = exg_error(env, "allow_groups must be 0 or 1");
    goto END;
  }
  handle = *resource;
  int index_count = (int)(bin.size / sizeof(int));
  for (bst_ulong i = 0; i < index_count; i++) {
    if (((int *)bin.data)[i] < 0) {
      ret = exg_error(env, "Indices must be non-negative");
      goto END;
    }
  }
  result = XGDMatrixSliceDMatrixEx(handle, (int *)bin.data, index_count, &out,
                                   allow_groups);
  if (0 == result) {
    ret = make_DMatrix_resource(env, out);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  return ret;
}

ERL_NIF_TERM EXGProxyDMatrixCreate(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  int result = -1;
  ERL_NIF_TERM ret = -1;
  if (argc != 0) {
    ret = exg_error(env, "EXGProxyDMatrixCreate doesn't take arguments");
    goto END;
  }
  result = XGProxyDMatrixCreate(&handle);
  if (0 == result) {
    ret = make_DMatrix_resource(env, handle);
  } else {
    ret = exg_error(env, XGBGetLastError());
  }
END:
  return ret;
}

// Helper to parse ArrayInterface JSON and copy data atomically
static int exg_parse_and_copy_array_interface(
    ErlNifEnv *env,
    const char *json_str,
    ERL_NIF_TERM *out_map
) {
  // Parse the JSON to extract address, typestr, shape
  // For now, we'll use a simple parser since the format is predictable:
  // {"version":3,"typestr":"<u8","data":[address,true],"shape":[n]}

  const char *data_start = strstr(json_str, "\"data\":[");
  const char *shape_start = strstr(json_str, "\"shape\":[");
  const char *typestr_start = strstr(json_str, "\"typestr\":\"");

  if (!data_start || !shape_start || !typestr_start) {
    return 0;
  }

  // Parse address from data array
  uintptr_t address = 0;
  if (sscanf(data_start, "\"data\":[%" SCNuPTR, &address) != 1) {
    return 0;
  }

  // Parse typestr (e.g., "<u8" or "<f4")
  char typestr[16] = {0};
  if (sscanf(typestr_start, "\"typestr\":\"%15[^\"]\""  , typestr) != 1) {
    return 0;
  }

  // Parse shape - for simplicity, handle single dimension [n]
  size_t shape_len = 0;
  if (sscanf(shape_start, "\"shape\":[%zu]"  , &shape_len) != 1) {
    return 0;
  }

  // Calculate size from typestr and shape
  // typestr format: <endian><type><bytes> e.g. "<u8" = 8 bytes, "<f4" = 4 bytes
  size_t bytes_per_elem = 0;
  if (sscanf(typestr + 2, "%zu"  , &bytes_per_elem) != 1) {
    return 0;
  }

  // Check for overflow
  if (shape_len > SIZE_MAX / bytes_per_elem) {
    return 0;
  }
  size_t total_size = shape_len * bytes_per_elem;

  // Copy data from XGBoost-owned address immediately using enif_make_new_binary
  // This creates the binary term directly, so ownership transfers to BEAM immediately
  if (address == 0 && total_size != 0) {
    return 0;
  }

  ERL_NIF_TERM data_binary_term;
  unsigned char *data_dest = enif_make_new_binary(env, total_size, &data_binary_term);
  if (data_dest == NULL && total_size != 0) {
    return 0;
  }
  if (total_size != 0) {
    memcpy(data_dest, (const void *)address, total_size);
  }

  // Convert typestr to binary string using enif_make_new_binary
  size_t typestr_len = strlen(typestr);
  ERL_NIF_TERM typestr_binary_term;
  unsigned char *typestr_dest = enif_make_new_binary(env, typestr_len, &typestr_binary_term);
  if (typestr_dest == NULL && typestr_len != 0) {
    return 0;
  }
  memcpy(typestr_dest, typestr, typestr_len);

  // Build result map: %{binary: binary, typestr: string, shape: [n]}
  ERL_NIF_TERM binary_key = enif_make_atom(env, "binary");
  ERL_NIF_TERM typestr_key = enif_make_atom(env, "typestr");
  ERL_NIF_TERM shape_key = enif_make_atom(env, "shape");

  ERL_NIF_TERM keys[] = {binary_key, typestr_key, shape_key};
  ERL_NIF_TERM values[] = {
    data_binary_term,
    typestr_binary_term,
    enif_make_list1(env, enif_make_uint64(env, (ErlNifUInt64)shape_len))
  };

  return enif_make_map_from_arrays(env, keys, values, 3, out_map);
}

ERL_NIF_TERM EXGDMatrixGetQuantileCut(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  DMatrixHandle handle;
  DMatrixHandle **resource = NULL;
  char *config = NULL;
  char const *out_indptr_json = NULL;
  char const *out_data_json = NULL;
  ERL_NIF_TERM indptr_map, data_map;
  ERL_NIF_TERM ret = -1;
  int result = -1;

  if (argc != 2) {
    ret = exg_error(env, "Wrong number of arguments");
    goto END;
  }
  if (!enif_get_resource(env, argv[0], DMatrix_RESOURCE_TYPE, (void *)&resource)) {
    ret = exg_error(env, "DMatrix must be a resource");
    goto END;
  }
  if (!exg_get_string(env, argv[1], &config)) {
    ret = exg_error(env, "Config must be a JSON-Encoded string");
    goto END;
  }
  handle = *resource;

  // Get ArrayInterface JSON from XGBoost
  result = XGDMatrixGetQuantileCut(handle, config, &out_indptr_json, &out_data_json);
  if (result != 0) {
    ret = exg_error(env, XGBGetLastError());
    goto END;
  }

  // Parse JSON and copy data atomically, before any other operations
  if (!exg_parse_and_copy_array_interface(env, out_indptr_json, &indptr_map)) {
    ret = exg_error(env, "Failed to parse indptr ArrayInterface");
    goto END;
  }
  if (!exg_parse_and_copy_array_interface(env, out_data_json, &data_map)) {
    ret = exg_error(env, "Failed to parse data ArrayInterface");
    goto END;
  }

  ret = exg_ok(env, enif_make_tuple2(env, indptr_map, data_map));

END:
  if (config != NULL) {
    enif_free(config);
    config = NULL;
  }
  return ret;
}
