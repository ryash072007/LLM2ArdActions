from django.shortcuts import render
from rest_framework.views import APIView
import wifi.ollama_interactions as OLLAMA
from rest_framework.response import Response

ollama = OLLAMA.OllamaInteractions("gemma2:2b")


class GetActionFromPrompt(APIView):
    def get(self, request):
        prompt = request.data.get("prompt")
        if prompt is None:
            return Response({"error": "prompt is required"}, status=400)
        print("Prompt: ", prompt)
        response = ollama.get_response(prompt)
        print(response)
        return Response({"response": response})
