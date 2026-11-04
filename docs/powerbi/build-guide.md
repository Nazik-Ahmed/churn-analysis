# Power BI build guide

Four pages built on one flat table (`Subs`). Mockups are in [`../images`](../images).

## Setup order
1. **Power Query:** paste [`powerquery.m`](powerquery.m) into Home > Advanced Editor (clear the editor first). Name the query `Subs`, then Close & Apply.
2. **Checks:** 2,400 rows; `ChurnFlag` sums to 223; `OnboardedFlag` sums to 1,314.
3. **Columns:** add [`calculated_columns.dax`](calculated_columns.dax) to `Subs`, then set Sort by column for the band columns.
4. **Measures:** add [`measures.dax`](measures.dax) to a `_Measures` table. Set formats (rates as Percentage, 1 decimal).
5. Set `ChurnFlag`, `OnboardedFlag`, tickets, sessions, discount, tenure and price to **Don't summarize**.

Expected check values: Total 2,400 · Churned 223 · Retained 2,177 · Churn rate 9.3% · average sessions 27.3 (churned) vs 30.5 (retained) · average tickets 4.04 vs 2.66.

## Pages
| Page | Visuals |
|---|---|
| 1. Executive Overview | 4 cards (Total, Churned, Churn Rate %, Retention Rate %); churn rate by package tier, billing cycle and onboarding; tier × cycle matrix with gradient background |
| 2. Behavioral Analysis | Churn rate by SessionBand and TicketBand with error bars; average sessions and tickets, churned vs retained; TicketBand × SessionBand matrix |
| 3. Segments & Risk | AtRisk vs Other table with Wilson bounds; churn by tier and AtRisk; payment method and discount charts (shown as non-drivers) |
| 4. Key Drivers | Key influencers visual |

Slicers: PackageTier, BillingCycle, PaymentMethod, AttendedOnboarding, SubscriptionDate (Between). Sync them via View > Sync slicers.

## Details that are easy to get wrong
- **Error bars** are in the **Analytics** pane (not Format). Choose *By field*, upper bound = `Wilson High`, lower bound = `Wilson Low`, relation **Absolute**, symmetrical **off**. Supported on clustered column/bar and line charts.
- **Reference line at the overall rate:** use a Line and clustered column chart with `Overall Churn Rate %` as the line (it ignores slicers), or a fixed Y-axis constant line. The Average line is not the overall rate.
- **Matrix heatmap:** Cell elements > Background color > fx > Gradient, based on `Churn Rate %`, fixed min 0.05 and max 0.13. Treat cells with fewer than about 30 subscribers as unreliable.
- **Key influencers:** Analyze = `ChurnStatus` ("Churned"); Explain by = SupportTickets, MonthlySessions, AttendedOnboarding, BillingCycle, PackageTier, PaymentMethod, DiscountPct, TenureMonths. Leave out MonthlyPrice (duplicates plan), IDs and the derived band columns. It ranks factors but shows no confidence intervals; use pages 2 and 3 and the Python results for those.
- The date slicer filters **signup** date. The data has no cancellation date.
