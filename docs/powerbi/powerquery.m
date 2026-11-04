// Power Query (Home > Advanced Editor). Replace the file path.
// Clear the editor completely before pasting, or you get "Token Eof expected".
let
    Source = Csv.Document(
        File.Contents("C:\Data\subscriptions-churn.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Renamed = Table.RenameColumns(Promoted, {
        {"معرف المشترك", "SubscriberID"},
        {"تاريخ الاشتراك", "SubscriptionDate"},
        {"الباقة", "PackageTier"},
        {"السعر الشهري", "MonthlyPrice"},
        {"دورة الفوترة", "BillingCycle"},
        {"أشهر البقاء", "TenureMonths"},
        {"وسيلة الدفع", "PaymentMethod"},
        {"تذاكر الدعم", "SupportTickets"},
        {"متوسط الجلسات الشهرية", "MonthlySessions"},
        {"نسبة الخصم", "DiscountPct"},
        {"حضر التهيئة", "AttendedOnboarding"},
        {"ألغى الاشتراك", "Churned"}}),
    Trimmed = Table.TransformColumns(Renamed, {
        {"SubscriberID", Text.Trim, type text},
        {"PackageTier", Text.Trim, type text},
        {"BillingCycle", Text.Trim, type text},
        {"PaymentMethod", Text.Trim, type text},
        {"AttendedOnboarding", Text.Trim, type text},
        {"Churned", Text.Trim, type text}}),
    Typed = Table.TransformColumnTypes(Trimmed, {
        {"SubscriptionDate", type date},
        {"MonthlyPrice", Currency.Type},
        {"TenureMonths", Int64.Type},
        {"SupportTickets", Int64.Type},
        {"MonthlySessions", Int64.Type},
        {"DiscountPct", Int64.Type}}, "en-US"),
    ChurnFlag = Table.AddColumn(Typed, "ChurnFlag",
        each if [Churned] = "نعم" then 1 else if [Churned] = "لا" then 0 else null, Int64.Type),
    OnboardedFlag = Table.AddColumn(ChurnFlag, "OnboardedFlag",
        each if [AttendedOnboarding] = "نعم" then 1 else if [AttendedOnboarding] = "لا" then 0 else null, Int64.Type)
in
    OnboardedFlag
