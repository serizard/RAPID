import React, { useEffect, useState } from 'react';
import OpenAI from 'openai';

interface RiskAssessmentProps {
  diagnosisData: {
    name: string;
    value: number;
    color: string;
  }[];
}

interface AssessmentResult {
  currentStatus: {
    description: string;
    details: string;
    mainAffectedAreas: string[];
  };
  riskLevel: {
    percentage: number; // 0~100 사이의 위험도
    description: string;
    recommendations: string[];
    immediateActions: string[];
  };
}

const openai = new OpenAI({
  apiKey: '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', // 여기에 실제 OpenAI API 키를 입력하세요
  dangerouslyAllowBrowser: true,
});

const constructPrompt = (diagnosisData: RiskAssessmentProps['diagnosisData']) => {
  return `As a highly experienced neurologist specializing in aphasia, provide a comprehensive analysis of the following diagnosis results:

Diagnosis Values:
- Broca's Aphasia: ${diagnosisData[0].value}
- Wernicke's Aphasia: ${diagnosisData[1].value}
- Anomic Aphasia: ${diagnosisData[2].value}
- Control (Normal): ${diagnosisData[3].value}

### Risk Evaluation:
1. Use the "Control (Normal)" value as a baseline for healthy language function.
2. Compare the other values (Broca's, Wernicke's, Anomic Aphasia) to the "Control" value.
3. Assess the relative proportions of each aphasia type to the "Control" value to determine the overall severity of language impairment.
4. Calculate a comprehensive risk percentage (0-100) using these relationships:
   - 75-100: Critical risk level requiring immediate medical attention.
   - 50-74: High risk level requiring prompt intervention.
   - 25-49: Moderate risk level requiring regular monitoring.
   - 0-24: Low risk level requiring periodic check-ups.

### Current Status Analysis:
- Provide a description of the patient's current condition.
- Highlight the main affected areas of language and communication.
- Discuss potential impacts on daily life.

### Recommendations:
- List immediate actions based on the risk level.
- Provide long-term management recommendations.

### Response Format:
{
  "currentStatus": {
    "description": "Concise summary of current condition",
    "details": "Detailed explanation of the aphasia type and its manifestations",
    "mainAffectedAreas": ["list", "of", "affected", "areas"]
  },
  "riskLevel": {
    "percentage": "number between 0-100",
    "description": "Detailed explanation of the risk level and its implications",
    "recommendations": ["list", "of", "recommended", "actions"],
    "immediateActions": ["list", "of", "immediate", "steps", "needed"]
  }
}`;
};

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ diagnosisData }) => {
  const [assessment, setAssessment] = useState<AssessmentResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAssessment = async () => {
      try {
        const completion = await openai.chat.completions.create({
          messages: [
            {
              role: "system",
              content:
                "You are an expert neurologist with extensive experience in aphasia diagnosis and treatment. Provide detailed, professional assessments while maintaining clear and actionable recommendations.",
            },
            {
              role: "user",
              content: constructPrompt(diagnosisData),
            },
          ],
          model: "gpt-3.5-turbo",
        });

        const content = completion.choices[0].message.content;
        if (!content) {
          throw new Error('No content received from OpenAI');
        }

        const result = JSON.parse(content) as AssessmentResult;
        setAssessment(result);
      } catch (error) {
        console.error('Error fetching assessment:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAssessment();
  }, [diagnosisData]);

  if (loading) {
    return (
      <div className="bg-white shadow-lg rounded-lg p-6">
        <h2 className="text-xl text-blue-600 font-bold mb-4">Risk Assessment</h2>
        <div className="flex justify-center items-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white shadow-lg rounded-lg p-6">
      <h2 className="text-xl text-blue-600 font-bold mb-4">Risk Assessment</h2>
      <div className="space-y-4">
        {/* Current Status Section */}
        <div className="p-4 bg-yellow-50 rounded-lg">
          <h4 className="font-bold">Current Status</h4>
          <p className="text-sm text-gray-600 mt-2">
            {assessment?.currentStatus.description}
          </p>
          <div className="mt-3">
            <p className="text-sm text-gray-600">{assessment?.currentStatus.details}</p>
          </div>
          {assessment?.currentStatus.mainAffectedAreas && (
            <div className="mt-3">
              <p className="text-sm font-semibold">Main Affected Areas:</p>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {assessment.currentStatus.mainAffectedAreas.map((area, index) => (
                  <li key={index}>{area}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Risk Level Section */}
        <div className="p-4 bg-yellow-50 rounded-lg">
          <h4 className="font-bold">Risk Level</h4>
          <p className="text-sm text-gray-600 mt-2">
            {assessment?.riskLevel?.description || 'No description available'}
          </p>
          <div className="w-full bg-gray-200 rounded h-2 mt-3">
            <div
              className={`h-2 rounded transition-all duration-500 ${
                assessment?.riskLevel?.percentage !== undefined
                  ? assessment.riskLevel.percentage >= 75
                    ? 'bg-red-500'
                    : assessment.riskLevel.percentage >= 50
                    ? 'bg-orange-500'
                    : assessment.riskLevel.percentage >= 25
                    ? 'bg-yellow-500'
                    : 'bg-green-500'
                  : 'bg-gray-300'
              }`}
              style={{
                width: `${assessment?.riskLevel?.percentage || 0}%`,
              }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-1 text-right">
            Risk Level: {assessment?.riskLevel?.percentage !== undefined ? `${assessment.riskLevel.percentage}%` : 'N/A'}
          </p>

          {assessment?.riskLevel?.recommendations && (
            <div className="mt-4">
              <p className="text-sm font-semibold">Recommendations:</p>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {assessment.riskLevel.recommendations.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
          )}

          {assessment?.riskLevel?.immediateActions && (
            <div className="mt-3">
              <p className="text-sm font-semibold">Immediate Actions Needed:</p>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {assessment.riskLevel.immediateActions.map((action, index) => (
                  <li key={index}>{action}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment;
