# Validate the JSON response which is converted from Natural
# Language input provided by user.

class SalesInputValidator:

    REQUIRED_FIELDS = [
        "marketing_spend",
        "customers",
        "orders",
        "holiday",
        "price"
    ]

    def validate(self, data):

        missing_fields = []

        for field in self.REQUIRED_FIELDS:

            if field not in data or data[field] is None:
                missing_fields.append(field)

        if missing_fields:

            raise ValueError(
                f"Missing fields: {missing_fields}"
            )

        if data["marketing_spend"] < 0:
            raise ValueError(
                "Marketing spend cannot be negative"
            )

        if data["customers"] < 0:
            raise ValueError(
                "Customers cannot be negative"
            )

        if data["orders"] < 0:
            raise ValueError(
                "Orders cannot be negative"
            )

        if data["price"] <= 0:
            raise ValueError(
                "Price must be greater than zero"
            )

        if data["holiday"] not in [0, 1]:
            raise ValueError(
                "Holiday must be 0 or 1"
            )

        return True