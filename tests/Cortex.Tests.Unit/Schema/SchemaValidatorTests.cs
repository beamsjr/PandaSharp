using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Schema;

namespace Cortex.Tests.Unit.Schema;

public class SchemaValidatorTests
{
    private DataFrame CreateTestDF() => new(
        new Column<int>("id", [1, 2, 3]),
        new Column<double>("score", [85.5, 92.0, 78.3]),
        new StringColumn("name", ["Alice", "Bob", "Charlie"]),
        new StringColumn("status", ["active", "inactive", "active"])
    );

    // ===== Column type validation =====

    [Fact]
    public void Validate_CorrectTypes_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Column("id", type: typeof(int))
            .Column("score", type: typeof(double))
            .Column("name", type: typeof(string));

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_WrongType_Fails()
    {
        var schema = DataFrameSchema.Define()
            .Column("id", type: typeof(double)); // wrong — it's int

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("Expected type Double");
    }

    // ===== Null validation =====

    [Fact]
    public void Validate_NoNulls_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Column("id", noNulls: true);

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_NoNulls_FailsWhenNullPresent()
    {
        var df = new DataFrame(Column<int>.FromNullable("x", [1, null, 3]));
        var schema = DataFrameSchema.Define()
            .Column("x", noNulls: true);

        var result = schema.Validate(df);
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("null values");
    }

    // ===== Uniqueness =====

    [Fact]
    public void Validate_Unique_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Column("id", unique: true);

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_Unique_FailsWithDuplicates()
    {
        var df = new DataFrame(new Column<int>("x", [1, 2, 1]));
        var schema = DataFrameSchema.Define()
            .Column("x", unique: true);

        var result = schema.Validate(df);
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("duplicate");
    }

    // ===== Min/Max range =====

    [Fact]
    public void Validate_Range_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Column("score", min: 0, max: 100);

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_BelowMin_Fails()
    {
        var df = new DataFrame(new Column<double>("x", [5, -1, 10]));
        var schema = DataFrameSchema.Define()
            .Column("x", min: 0);

        var result = schema.Validate(df);
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("below minimum");
    }

    [Fact]
    public void Validate_AboveMax_Fails()
    {
        var df = new DataFrame(new Column<double>("x", [5, 10, 200]));
        var schema = DataFrameSchema.Define()
            .Column("x", max: 100);

        var result = schema.Validate(df);
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("above maximum");
    }

    // ===== Regex pattern =====

    [Fact]
    public void Validate_Pattern_Passes()
    {
        var df = new DataFrame(
            new StringColumn("email", ["a@b.com", "x@y.org"])
        );
        var schema = DataFrameSchema.Define()
            .Column("email", pattern: @".+@.+\..+");

        schema.Validate(df).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_Pattern_Fails()
    {
        var df = new DataFrame(
            new StringColumn("email", ["a@b.com", "invalid"])
        );
        var schema = DataFrameSchema.Define()
            .Column("email", pattern: @".+@.+\..+");

        var result = schema.Validate(df);
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("doesn't match pattern");
    }

    // ===== Allowed values =====

    [Fact]
    public void Validate_AllowedValues_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Column("status", allowedValues: ["active", "inactive"]);

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_AllowedValues_Fails()
    {
        var schema = DataFrameSchema.Define()
            .Column("status", allowedValues: ["active"]); // "inactive" not allowed

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("not in allowed values");
    }

    // ===== HasColumns =====

    [Fact]
    public void Validate_HasColumns_Passes()
    {
        var schema = DataFrameSchema.Define()
            .HasColumns("id", "score", "name");

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_HasColumns_MissingColumn_Fails()
    {
        var schema = DataFrameSchema.Define()
            .HasColumns("id", "nonexistent");

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("Missing required columns");
    }

    // ===== NoExtraColumns =====

    [Fact]
    public void Validate_NoExtraColumns_Fails()
    {
        var schema = DataFrameSchema.Define()
            .Column("id")
            .NoExtraColumns();

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("Unexpected columns");
    }

    // ===== MinRows / MaxRows =====

    [Fact]
    public void Validate_MinRows_Passes()
    {
        var schema = DataFrameSchema.Define().MinRows(1);
        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_MinRows_Fails()
    {
        var schema = DataFrameSchema.Define().MinRows(10);
        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
    }

    [Fact]
    public void Validate_MaxRows_Fails()
    {
        var schema = DataFrameSchema.Define().MaxRows(2);
        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
    }

    // ===== NoDuplicateRows =====

    [Fact]
    public void Validate_NoDuplicateRows_Passes()
    {
        var schema = DataFrameSchema.Define().NoDuplicateRows();
        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_NoDuplicateRows_Fails()
    {
        var df = new DataFrame(
            new Column<int>("x", [1, 2, 1]),
            new StringColumn("y", ["a", "b", "a"])
        );
        var schema = DataFrameSchema.Define().NoDuplicateRows();
        schema.Validate(df).IsValid.Should().BeFalse();
    }

    // ===== Custom check =====

    [Fact]
    public void Validate_CustomCheck_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Check("has_rows", df => df.RowCount > 0);

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue();
    }

    [Fact]
    public void Validate_CustomCheck_Fails()
    {
        var schema = DataFrameSchema.Define()
            .Check("too_few", df => df.RowCount >= 100, "Need at least 100 rows");

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("Need at least 100 rows");
    }

    // ===== Custom column check =====

    [Fact]
    public void Validate_CustomColumnCheck_Passes()
    {
        var schema = DataFrameSchema.Define()
            .Column("score", check: col =>
            {
                for (int i = 0; i < col.Length; i++)
                    if (!col.IsNull(i) && Convert.ToDouble(col.GetObject(i)) < 70)
                        return new ValidationError(col.Name, "All scores must be >= 70");
                return null;
            });

        schema.Validate(CreateTestDF()).IsValid.Should().BeTrue(); // all scores >= 70
    }

    [Fact]
    public void Validate_CustomColumnCheck_Fails()
    {
        var df = new DataFrame(new Column<double>("score", [90, 50, 80]));
        var schema = DataFrameSchema.Define()
            .Column("score", check: col =>
            {
                for (int i = 0; i < col.Length; i++)
                    if (!col.IsNull(i) && Convert.ToDouble(col.GetObject(i)) < 70)
                        return new ValidationError(col.Name, $"Score {col.GetObject(i)} at index {i} < 70");
                return null;
            });

        var result = schema.Validate(df);
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("50");
    }

    // ===== Missing column =====

    [Fact]
    public void Validate_MissingColumn_ReportsError()
    {
        var schema = DataFrameSchema.Define()
            .Column("nonexistent", type: typeof(int));

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeFalse();
        result.Errors[0].Message.Should().Contain("not found");
    }

    // ===== Multiple errors =====

    [Fact]
    public void Validate_MultipleErrors_AllReported()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("x", [1, null, 1])
        );
        var schema = DataFrameSchema.Define()
            .Column("x", noNulls: true, unique: true);

        var result = schema.Validate(df);
        result.ErrorCount.Should().Be(2); // null + duplicate
    }

    // ===== ValidateSchema extension =====

    [Fact]
    public void ValidateSchema_PassesThrough()
    {
        var schema = DataFrameSchema.Define()
            .Column("id", type: typeof(int));

        var df = CreateTestDF().ValidateSchema(schema);
        df.RowCount.Should().Be(3); // returns same DataFrame
    }

    [Fact]
    public void ValidateSchema_ThrowsOnFailure()
    {
        var schema = DataFrameSchema.Define()
            .Column("id", type: typeof(string));

        var act = () => CreateTestDF().ValidateSchema(schema);
        act.Should().Throw<InvalidDataException>();
    }

    // ===== ToString =====

    [Fact]
    public void ValidationResult_ToString_Valid()
    {
        var result = DataFrameSchema.Define().Validate(CreateTestDF());
        result.ToString().Should().Contain("passed");
    }

    [Fact]
    public void ValidationResult_ToString_Invalid()
    {
        var schema = DataFrameSchema.Define().MinRows(100);
        var result = schema.Validate(CreateTestDF());
        result.ToString().Should().Contain("failed");
    }

    // ===== Comprehensive schema =====

    [Fact]
    public void Validate_ComprehensiveSchema()
    {
        var schema = DataFrameSchema.Define()
            .HasColumns("id", "score", "name", "status")
            .Column("id", type: typeof(int), unique: true, noNulls: true)
            .Column("score", type: typeof(double), min: 0, max: 100)
            .Column("name", type: typeof(string), noNulls: true)
            .Column("status", allowedValues: ["active", "inactive"])
            .MinRows(1)
            .NoDuplicateRows();

        var result = schema.Validate(CreateTestDF());
        result.IsValid.Should().BeTrue();
    }
}
