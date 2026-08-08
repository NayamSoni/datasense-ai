def get_schema(df):

    schema = []

    for col in df.columns:

        schema.append(
            {
                "name": col,
                "dtype": str(df[col].dtype)
            }
        )

    return schema