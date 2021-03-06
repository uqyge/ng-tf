import { BrowserModule } from "@angular/platform-browser";
import { NgModule } from "@angular/core";

import { AppRoutingModule } from "./app-routing.module";
import { AppComponent } from "./app.component";
import { WebcamModule } from "./modules/webcam/webcam.module";
import { MnetComponent } from './mnet/mnet.component';

@NgModule({
  declarations: [AppComponent, MnetComponent],
  imports: [WebcamModule, BrowserModule, AppRoutingModule],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
