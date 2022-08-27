
   const ttmForm = get(".ttm-loginarea");
   const ttmInput = get(".ttm-input");
   const userNameInput = get(".username-input");
   const ttmButton = get(".ttm-send-btn");
   ttmForm.addEventListener("submit", event => {
    event.preventDefault();
    const msgText = ttmInput.value;
    const username = userNameInput.value;
    if (!msgText) {
      ttmInput.placeholder = "Please enter the password!"
      return;
    }
    if (!username) {
      userNameInput.placeholder = "Please enter a username!"
      return;
    }
    ttmButton.innerHTML = "Loading..."
    pwApprove(msgText, username, pwApprovalPath);
  });

  function pwApprove(rawText, username, pw_url) {
    // Bot Response
    result = $.ajax({
        type: 'POST',
          url: pw_url,
          data: JSON.stringify({password : rawText, user: username}),
          processData: false,
          contentType: false,
          cache: false,
          success : function(text)
            {
              if (text == "denied") {
                ttmButton.innerHTML = "🔓";
                $(document).ready(function() { $(".ttm-loginarea").effect( "shake", {times:4}, 250 ) });
              } else if (text == "notify") {
                ttmButton.innerHTML = "🔓";
                $(document).ready(function() { $(".ttm-loginarea").effect( "shake", {times:4}, 250 ) });
                alert("The password has changed. Please reach out to Dylan (dslack@uci.edu) for new password. He has been notified of this attempt and will try to reach out to you as well if he can identify your username.")
              } else {
                response = text;
                console.log(response);
                window.location.replace(response);
              }
            }
        });
  }

  // Utils
    function get(selector, root = document) {
        return root.querySelector(selector);
      }
