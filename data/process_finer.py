import json
import random

from datasets import load_dataset
import jsonlines
from tqdm import tqdm

labels = _LABELS = [
    "-",
    "B-AccrualForEnvironmentalLossContingencies",
    "B-AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife",
    "I-AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife",
    "B-AllocatedShareBasedCompensationExpense",
    "B-AmortizationOfFinancingCosts",
    "B-AmortizationOfIntangibleAssets",
    "I-AmortizationOfIntangibleAssets",
    "B-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount",
    "I-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount",
    "B-AreaOfRealEstateProperty",
    "I-AreaOfRealEstateProperty",
    "B-AssetImpairmentCharges",
    "B-BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued",
    "B-BusinessAcquisitionPercentageOfVotingInterestsAcquired",
    "I-BusinessAcquisitionPercentageOfVotingInterestsAcquired",
    "B-BusinessCombinationAcquisitionRelatedCosts",
    "B-BusinessCombinationConsiderationTransferred1",
    "B-BusinessCombinationContingentConsiderationLiability",
    "B-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill",
    "B-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles",
    "B-CapitalizedContractCostAmortization",
    "B-CashAndCashEquivalentsFairValueDisclosure",
    "B-ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1",
    "B-CommonStockCapitalSharesReservedForFutureIssuance",
    "B-CommonStockDividendsPerShareDeclared",
    "B-CommonStockParOrStatedValuePerShare",
    "B-CommonStockSharesAuthorized",
    "I-CommonStockSharesAuthorized",
    "B-CommonStockSharesOutstanding",
    "B-ConcentrationRiskPercentage1",
    "B-ContractWithCustomerLiability",
    "B-ContractWithCustomerLiabilityRevenueRecognized",
    "B-CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption",
    "B-DebtInstrumentBasisSpreadOnVariableRate1",
    "B-DebtInstrumentCarryingAmount",
    "B-DebtInstrumentConvertibleConversionPrice1",
    "B-DebtInstrumentFaceAmount",
    "I-DebtInstrumentFaceAmount",
    "B-DebtInstrumentFairValue",
    "B-DebtInstrumentInterestRateEffectivePercentage",
    "B-DebtInstrumentInterestRateStatedPercentage",
    "B-DebtInstrumentMaturityDate",
    "I-DebtInstrumentMaturityDate",
    "B-DebtInstrumentRedemptionPricePercentage",
    "B-DebtInstrumentTerm",
    "I-DebtInstrumentTerm",
    "B-DebtInstrumentUnamortizedDiscount",
    "B-DebtWeightedAverageInterestRate",
    "B-DeferredFinanceCostsGross",
    "B-DeferredFinanceCostsNet",
    "B-DefinedBenefitPlanContributionsByEmployer",
    "B-DefinedContributionPlanCostRecognized",
    "B-Depreciation",
    "B-DerivativeFixedInterestRate",
    "B-DerivativeNotionalAmount",
    "B-DisposalGroupIncludingDiscontinuedOperationConsideration",
    "B-EffectiveIncomeTaxRateContinuingOperations",
    "B-EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate",
    "B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
    "B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1",
    "I-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1",
    "B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions",
    "B-EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense",
    "B-EquityMethodInvestmentOwnershipPercentage",
    "I-EquityMethodInvestmentOwnershipPercentage",
    "B-EquityMethodInvestments",
    "B-FiniteLivedIntangibleAssetUsefulLife",
    "I-FiniteLivedIntangibleAssetUsefulLife",
    "B-GainsLossesOnExtinguishmentOfDebt",
    "B-Goodwill",
    "B-GoodwillImpairmentLoss",
    "B-GuaranteeObligationsMaximumExposure",
    "B-IncomeLossFromEquityMethodInvestments",
    "B-IncomeTaxExpenseBenefit",
    "B-InterestExpense",
    "B-InterestExpenseDebt",
    "B-LeaseAndRentalExpense",
    "B-LesseeOperatingLeaseRenewalTerm",
    "I-LesseeOperatingLeaseRenewalTerm",
    "B-LesseeOperatingLeaseTermOfContract",
    "I-LesseeOperatingLeaseTermOfContract",
    "B-LettersOfCreditOutstandingAmount",
    "B-LineOfCredit",
    "B-LineOfCreditFacilityCommitmentFeePercentage",
    "B-LineOfCreditFacilityCurrentBorrowingCapacity",
    "B-LineOfCreditFacilityInterestRateAtPeriodEnd",
    "B-LineOfCreditFacilityMaximumBorrowingCapacity",
    "B-LineOfCreditFacilityRemainingBorrowingCapacity",
    "B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage",
    "B-LongTermDebt",
    "B-LongTermDebtFairValue",
    "B-LossContingencyAccrualAtCarryingValue",
    "B-LossContingencyDamagesSoughtValue",
    "B-LossContingencyEstimateOfPossibleLoss",
    "B-LossContingencyPendingClaimsNumber",
    "I-LossContingencyPendingClaimsNumber",
    "B-MinorityInterestOwnershipPercentageByNoncontrollingOwners",
    "B-MinorityInterestOwnershipPercentageByParent",
    "B-NumberOfOperatingSegments",
    "B-NumberOfRealEstateProperties",
    "I-NumberOfRealEstateProperties",
    "B-NumberOfReportableSegments",
    "B-OperatingLeaseCost",
    "B-OperatingLeaseExpense",
    "B-OperatingLeaseLiability",
    "B-OperatingLeasePayments",
    "B-OperatingLeaseRightOfUseAsset",
    "B-OperatingLeaseWeightedAverageDiscountRatePercent",
    "B-OperatingLeaseWeightedAverageRemainingLeaseTerm1",
    "I-OperatingLeaseWeightedAverageRemainingLeaseTerm1",
    "B-OperatingLeasesRentExpenseNet",
    "B-OperatingLossCarryforwards",
    "B-PaymentsToAcquireBusinessesGross",
    "B-PaymentsToAcquireBusinessesNetOfCashAcquired",
    "B-PreferredStockDividendRatePercentage",
    "B-PreferredStockSharesAuthorized",
    "I-PreferredStockSharesAuthorized",
    "B-ProceedsFromIssuanceOfCommonStock",
    "B-PropertyPlantAndEquipmentUsefulLife",
    "I-PropertyPlantAndEquipmentUsefulLife",
    "B-PublicUtilitiesRequestedRateIncreaseDecreaseAmount",
    "B-RelatedPartyTransactionAmountsOfTransaction",
    "I-RelatedPartyTransactionAmountsOfTransaction",
    "B-RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty",
    "I-RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty",
    "B-RepaymentsOfDebt",
    "B-RestructuringAndRelatedCostExpectedCost1",
    "B-RestructuringCharges",
    "B-RevenueFromContractWithCustomerExcludingAssessedTax",
    "B-RevenueFromContractWithCustomerIncludingAssessedTax",
    "B-RevenueFromRelatedParties",
    "B-RevenueRemainingPerformanceObligation",
    "B-Revenues",
    "B-SaleOfStockNumberOfSharesIssuedInTransaction",
    "I-SaleOfStockNumberOfSharesIssuedInTransaction",
    "B-SaleOfStockPricePerShare",
    "B-ShareBasedCompensation",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross",
    "B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue",
    "B-SharePrice",
    "B-SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage",
    "I-SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage",
    "B-SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod",
    "I-SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod",
    "B-StockIssuedDuringPeriodSharesNewIssues",
    "I-StockIssuedDuringPeriodSharesNewIssues",
    "B-StockRepurchaseProgramAuthorizedAmount1",
    "B-StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
    "B-StockRepurchasedAndRetiredDuringPeriodShares",
    "B-StockRepurchasedDuringPeriodShares",
    "I-StockRepurchasedDuringPeriodShares",
    "B-SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense",
    "B-TreasuryStockAcquiredAverageCostPerShare",
    "B-TreasuryStockSharesAcquired",
    "I-TreasuryStockSharesAcquired",
    "B-TreasuryStockValueAcquiredCostMethod",
    "B-UnrecognizedTaxBenefits",
    "B-UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate",
    "I-DeferredFinanceCostsGross",
    "I-CommonStockParOrStatedValuePerShare",
    "I-LossContingencyEstimateOfPossibleLoss",
    "I-DefinedContributionPlanCostRecognized",
    "I-DebtInstrumentFairValue",
    "I-ContractWithCustomerLiabilityRevenueRecognized",
    "I-RevenueRemainingPerformanceObligation",
    "I-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
    "I-DebtInstrumentInterestRateStatedPercentage",
    "I-OperatingLossCarryforwards",
    "I-MinorityInterestOwnershipPercentageByNoncontrollingOwners",
    "I-InterestExpense",
    "I-LongTermDebt",
    "I-ShareBasedCompensation",
    "I-DebtWeightedAverageInterestRate",
    "I-DebtInstrumentCarryingAmount",
    "I-DebtInstrumentConvertibleConversionPrice1",
    "I-IncomeTaxExpenseBenefit",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue",
    "I-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions",
    "I-EquityMethodInvestments",
    "I-DebtInstrumentUnamortizedDiscount",
    "I-GainsLossesOnExtinguishmentOfDebt",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant",
    "I-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill",
    "I-PreferredStockDividendRatePercentage",
    "I-RevenueFromContractWithCustomerIncludingAssessedTax",
    "I-OperatingLeaseWeightedAverageDiscountRatePercent",
    "I-LineOfCredit",
    "I-LineOfCreditFacilityMaximumBorrowingCapacity",
    "I-EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate",
    "I-LineOfCreditFacilityCommitmentFeePercentage",
    "I-BusinessCombinationConsiderationTransferred1",
    "I-CommonStockDividendsPerShareDeclared",
    "I-DebtInstrumentBasisSpreadOnVariableRate1",
    "I-DisposalGroupIncludingDiscontinuedOperationConsideration",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross",
    "I-CommonStockSharesOutstanding",
    "I-AmortizationOfFinancingCosts",
    "I-LineOfCreditFacilityCurrentBorrowingCapacity",
    "I-TreasuryStockValueAcquiredCostMethod",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber",
    "I-DebtInstrumentInterestRateEffectivePercentage",
    "I-SaleOfStockPricePerShare",
    "I-CapitalizedContractCostAmortization",
    "I-RestructuringCharges",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue",
    "I-AccrualForEnvironmentalLossContingencies",
    "I-CashAndCashEquivalentsFairValueDisclosure",
    "I-ProceedsFromIssuanceOfCommonStock",
    "I-Revenues",
    "I-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles",
    "I-LettersOfCreditOutstandingAmount",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue",
    "I-OperatingLeasePayments",
    "I-LineOfCreditFacilityRemainingBorrowingCapacity",
    "I-PaymentsToAcquireBusinessesGross",
    "I-TreasuryStockAcquiredAverageCostPerShare",
    "I-DeferredFinanceCostsNet",
    "I-StockRepurchaseProgramAuthorizedAmount1",
    "I-InterestExpenseDebt",
    "I-ContractWithCustomerLiability",
    "I-OperatingLeaseExpense",
    "I-Depreciation",
    "I-AllocatedShareBasedCompensationExpense",
    "I-LossContingencyAccrualAtCarryingValue",
    "I-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage",
    "I-SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense",
    "I-OperatingLeaseLiability",
    "I-RevenueFromRelatedParties",
    "I-PaymentsToAcquireBusinessesNetOfCashAcquired",
    "I-BusinessCombinationContingentConsiderationLiability",
    "I-LossContingencyDamagesSoughtValue",
    "I-NumberOfOperatingSegments",
    "I-BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued",
    "I-OperatingLeaseRightOfUseAsset",
    "I-BusinessCombinationAcquisitionRelatedCosts",
    "I-UnrecognizedTaxBenefits",
    "I-GuaranteeObligationsMaximumExposure",
    "I-RestructuringAndRelatedCostExpectedCost1",
    "I-DefinedBenefitPlanContributionsByEmployer",
    "I-OperatingLeaseCost",
    "I-DerivativeFixedInterestRate",
    "I-Goodwill",
    "I-GoodwillImpairmentLoss",
    "I-CommonStockCapitalSharesReservedForFutureIssuance",
    "I-StockRepurchasedAndRetiredDuringPeriodShares",
    "I-EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense",
    "I-IncomeLossFromEquityMethodInvestments",
    "I-NumberOfReportableSegments",
    "I-LongTermDebtFairValue",
    "I-RepaymentsOfDebt",
    "I-ConcentrationRiskPercentage1",
    "I-DebtInstrumentRedemptionPricePercentage",
    "I-CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption",
    "I-SharePrice",
    "I-UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate",
    "I-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue",
    "I-EffectiveIncomeTaxRateContinuingOperations",
    "I-RevenueFromContractWithCustomerExcludingAssessedTax",
    "I-StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
    "I-LineOfCreditFacilityInterestRateAtPeriodEnd",
    "I-ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1",
    "I-OperatingLeasesRentExpenseNet",
    "I-LeaseAndRentalExpense",
    "I-PublicUtilitiesRequestedRateIncreaseDecreaseAmount",
    "I-MinorityInterestOwnershipPercentageByParent",
    "I-AssetImpairmentCharges",
    "I-DerivativeNotionalAmount",
]
labels = [x.split("-")[1] for x in labels]
random.seed(42)
batched_prompt_size = 4

size_limit = {"train": 10000, "test": 10000}

def process_finer139_dataset(dataset_name):
    for split in ['train', 'test']:
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.shuffle(seed=42)
        batched_data = []
        current_batch_examples = []

        for example in tqdm(dataset, desc=f"Processing {split} split"):
            processed_examples = process_example(example, labels)
            if processed_examples:
                current_batch_examples.extend(processed_examples)

                while len(current_batch_examples) >= batched_prompt_size:
                    batch_to_process = current_batch_examples[:batched_prompt_size]
                    batched_data.append(create_batched_prompt(batch_to_process))
                    current_batch_examples = current_batch_examples[batched_prompt_size:]

            if len(batched_data)  > size_limit[split]:  # Adjust break condition for batched data count
                break

        # Process any remaining examples
        if current_batch_examples:
            batched_data.append(create_batched_prompt(current_batch_examples))

        print(len(batched_data))
        with open(f"{split}/finer_{split}_batched.jsonl", 'w') as writer:  # Changed filename to indicate batched data
            random.shuffle(batched_data)
            for x in batched_data:
                writer.write(json.dumps(x) + "\n")


def process_example(example, labels):
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    sentence = " ".join(tokens)

    ner_tag_idx = [tag_id for tag_id in ner_tags if tag_id != 0][:3]
    ner_tok_idx = [i for i in range(len(ner_tags)) if ner_tags[i] != 0][:3]

    results = []

    for i, x in zip(ner_tok_idx, ner_tag_idx):
        target = labels[x]
        if "-" in target:
            target = target.split("-")[1]

        processed_example = {
            "sentence": sentence,  # Keep sentence for batched prompt creation
            "word_index": i,  # Keep word index
            "word": tokens[i],  # Keep word
            "target": target,
        }
        results.append(processed_example)
    return results


def create_batched_prompt(batch_examples):
    prompt_questions = []
    targets = []

    for example in batch_examples:
        prompt_questions.append(f'What is best tag for entity "{example["word"]}" in sentence: "{example["sentence"]}?"')
        targets.append(example["target"])

    combined_prompt = (
            "You are XBRL expert.  "
            "Here is a list of US GAAP tags options: " + ",".join(list(set(labels))) + ". "
            f"Answer the following {batched_prompt_size} independent questions by providing only  {batched_prompt_size} US GAAP tags answers in the order of the questions. Each answer must be saperated by a comma (,).  Provide nothing else. \n" +
            "\n".join([f"{i + 1}. {question}" for i, question in enumerate(prompt_questions)]) +
            "\nOutput US GAAP tags:"
    )

    combined_target = ",".join(targets)

    batched_example = {
        "context": combined_prompt,
        "target": combined_target,
    }
    return batched_example


if __name__ == '__main__':
    dataset_name = "nlpaueb/finer-139"

    process_finer139_dataset(dataset_name)
