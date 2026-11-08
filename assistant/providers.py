import json
import os
import urllib.error
import urllib.request


TIMEOUT = 60


class ProviderError(Exception):
    pass


class ProviderUnavailable(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class NoProviderConfigured(ProviderError):
    pass


def _request(provider, prompt, history):
    api_key = os.environ.get(provider["key_env"], "").strip()
    if not api_key:
        raise NoProviderConfigured(provider["name"])

    model = configured_model(provider)
    messages = [{"role": "system", "content": (
        "Ты — Клодд, личный AI-ассистент. Общайся естественно и дружелюбно. "
        "Отвечай на языке пользователя, будь точным и не выдумывай факты. "
        "Пиши понятно, без лишних вступлений. "
        "Если сообщение содержит распознанный текст голосовой записи, воспринимай его как обычное сообщение пользователя. "
        "Не утверждай, что не умеешь принимать или распознавать голосовые сообщения: транскрипция уже выполнена до передачи текста тебе. "
        "Если пользователь просит повторить или проверить распознанное, выведи полученный текст без отказа. "
        f"Текущий основной AI-провайдер: {provider['name']}; активная модель: {model}. "
        "Если спросят о твоей модели или провайдере, используй только эти сведения. "
        "Codex — отдельный инструмент для технических задач и не является твоей моделью."
    )}]

    for item in history[-10:]:
        role = item.get("role")
        content = item.get("text", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }).encode()

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0.0",
    }

    if provider["name"] == "OpenRouter":
        headers["HTTP-Referer"] = "https://openrouter.ai"
        headers["X-Title"] = "Kлод"

    request = urllib.request.Request(
        provider["url"],
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            result = json.load(response)

        answer = result["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError("Провайдер вернул пустой ответ.")
        return answer.strip()

    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ProviderAuthenticationError(
                provider["name"] + ": ошибка авторизации API."
            ) from exc
        if exc.code in (408, 409, 425, 429) or exc.code >= 500:
            raise ProviderUnavailable(
                provider["name"] + ": временная ошибка API (" + str(exc.code) + ")."
            ) from exc
        raise ProviderError(
            provider["name"] + ": API вернул ошибку " + str(exc.code) + "."
        ) from exc

    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderUnavailable(
            provider["name"] + ": нет соединения или истёк таймаут."
        ) from exc

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError(
            provider["name"] + ": некорректный ответ API."
        ) from exc


PROVIDERS = (
    {
        "name": "Groq",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-20b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
    {
        "name": "OpenRouter",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "openrouter/free",
        "url": "https://openrouter.ai/api/v1/chat/completions",
    },
)


def configured_model(provider):
    """Return the model actually selected for this provider request."""
    return os.environ.get(provider["model_env"], "").strip() or provider["default_model"]


def configured_provider_models():
    """Expose active provider/model pairs without exposing credentials."""
    return tuple(
        (provider["name"], configured_model(provider))
        for provider in PROVIDERS
        if os.environ.get(provider["key_env"], "").strip()
    )


def generate_answer(prompt, history=None):
    history = history or []
    configured = [
        provider for provider in PROVIDERS
        if os.environ.get(provider["key_env"], "").strip()
    ]

    if not configured:
        raise NoProviderConfigured(
            "Не настроен ни один AI-провайдер. Нужен API-ключ Groq или OpenRouter."
        )

    errors = []
    authentication_errors = []

    for provider in configured:
        try:
            return _request(provider, prompt, history)
        except ProviderAuthenticationError as exc:
            errors.append(str(exc))
            authentication_errors.append(str(exc))
            continue
        except ProviderUnavailable as exc:
            errors.append(str(exc))
            continue

    if authentication_errors and len(authentication_errors) == len(errors):
        raise ProviderAuthenticationError(
            "Не удалось авторизоваться ни у одного настроенного AI-провайдера. "
            + " ".join(errors)
        )

    raise ProviderUnavailable(
        "Все настроенные AI-провайдеры временно недоступны. "
        + " ".join(errors)
    )


def configured_provider_names():
    """Return provider names only; never return or log credential values."""
    return tuple(
        provider["name"]
        for provider in PROVIDERS
        if os.environ.get(provider["key_env"], "").strip()
    )
